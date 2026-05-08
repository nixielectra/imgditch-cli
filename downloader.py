#!/usr/bin/env python3
"""
FileDitch Downloader — fast parallel chunk downloader
Writes each chunk directly to disk (no RAM bloat).

Usage:
    python fileditch_downloader.py <url> [options]

Examples:
    python fileditch_downloader.py "https://1.thegumonmyshoe.me/...mp4?md5=...&expires=..."
    python fileditch_downloader.py urls.txt --batch -d ./downloads
    python fileditch_downloader.py <url> --threads 16 --debug
"""

import argparse
import os
import re
import sys
import time
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

try:
    import requests
except ImportError:
    print("Missing dependency. Run: pip install requests")
    sys.exit(1)


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://fileditchfiles.me/",
}

CDN_PATTERN = re.compile(
    r'https?://[\w\-]+\.(?:thegumonmyshoe\.me|fileditchfiles\.me)'
    r'/[^\s"\'<>]+\?[^\s"\'<>]*(?:md5|expires)=[^\s"\'<>]+',
    re.IGNORECASE,
)

DEFAULT_THREADS = 2
READ_SIZE = 4 * 1024 * 1024   # 4 MB


# ── Progress ──────────────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int):
        self.total = total
        self.downloaded = 0
        self.lock = threading.Lock()
        self.start = time.time()

    def add(self, n: int):
        with self.lock:
            self.downloaded += n
            self._draw()

    def _draw(self):
        elapsed = max(time.time() - self.start, 0.001)
        speed = self.downloaded / elapsed
        spd = f"{speed/1024/1024:.1f} MB/s" if speed > 1024*1024 else f"{speed/1024:.0f} KB/s"
        if self.total:
            pct = self.downloaded / self.total * 100
            filled = int(30 * self.downloaded / self.total)
            bar = "█" * filled + "░" * (30 - filled)
            eta = (self.total - self.downloaded) / speed if speed else 0
            done_mb = self.downloaded / 1024 / 1024
            total_mb = self.total / 1024 / 1024
            print(f"\r  [{bar}] {pct:5.1f}%  {done_mb:.1f}/{total_mb:.1f} MB  {spd}  ETA {eta:.0f}s  ",
                  end="", flush=True)
        else:
            print(f"\r  {self.downloaded/1024/1024:.1f} MB  {spd}  ", end="", flush=True)

    def done(self):
        elapsed = max(time.time() - self.start, 0.001)
        avg = self.downloaded / elapsed
        spd = f"{avg/1024/1024:.1f} MB/s" if avg > 1024*1024 else f"{avg/1024:.0f} KB/s"
        mb = self.downloaded / 1024 / 1024
        print(f"\r  [{'█'*30}] 100.0%  {mb:.1f} MB  avg {spd}  {elapsed:.1f}s      ")


# ── URL helpers ───────────────────────────────────────────────────────────────

def extract_cdn_url(html: str, debug: bool = False) -> Optional[str]:
    matches = CDN_PATTERN.findall(html)
    if debug:
        print(f"  [parse] CDN matches: {matches}")
    if matches:
        return matches[0]
    fallback = re.findall(
        r'(?:href|src|url)[=:\s"\']+([^\s"\'<>]*(?:expires|md5)=[^\s"\'<>]+)',
        html, re.IGNORECASE)
    if fallback:
        u = fallback[0].strip("'\"")
        return ("https:" if u.startswith("//") else "https://") + u if not u.startswith("http") else u
    return None


def resolve_cdn_url(url: str, debug: bool = False) -> Optional[str]:
    print(f"  Fetching wrapper page...")
    r = requests.get(url, headers=BROWSER_HEADERS, allow_redirects=True, timeout=20)
    r.raise_for_status()
    if debug:
        print(f"  [page] Status={r.status_code}  len={len(r.text)}")
        print(f"  [page] HTML:\n{r.text[:1000]}\n---")
    cdn = extract_cdn_url(r.text, debug=debug)
    if cdn:
        print(f"  CDN URL: {cdn}")
    else:
        print("  [!] Could not find CDN URL. Use --debug to inspect HTML.")
    return cdn


def get_filename(url: str, response: Optional[requests.Response] = None,
                 original_url: str = "") -> str:
    if response:
        cd = response.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            fname = cd.split("filename=")[-1].strip().strip('"\'')
            if fname:
                return fname
    name = Path(urlparse(url).path).name
    if name and "." in name:
        return name
    if original_url:
        qs = parse_qs(urlparse(original_url).query)
        if "f" in qs:
            name = Path(qs["f"][0]).name
            if name:
                return name
    return "downloaded_file"


# ── Probe ─────────────────────────────────────────────────────────────────────

def probe(url: str, debug: bool = False) -> Tuple[int, bool]:
    """Return (total_bytes, accepts_ranges)."""
    try:
        r = requests.head(url, headers=DOWNLOAD_HEADERS, allow_redirects=True, timeout=15)
        cl = int(r.headers.get("Content-Length", 0))
        ar = "bytes" in r.headers.get("Accept-Ranges", "")
        if debug:
            print(f"  [probe] HEAD {r.status_code}  size={cl}  ranges={ar}")
        if cl:
            return cl, ar
    except Exception as e:
        if debug:
            print(f"  [probe] HEAD failed: {e}")

    try:
        h = {**DOWNLOAD_HEADERS, "Range": "bytes=0-0"}
        r = requests.get(url, headers=h, timeout=15)
        if r.status_code == 206:
            cr = r.headers.get("Content-Range", "")
            total = int(cr.split("/")[-1]) if "/" in cr else 0
            if debug:
                print(f"  [probe] Range 206  total={total}")
            return total, True
    except Exception as e:
        if debug:
            print(f"  [probe] Range probe failed: {e}")

    return 0, False


# ── Chunk download → temp file ────────────────────────────────────────────────

def download_chunk_to_file(url: str, start: int, end: int,
                           tmp_path: str, progress: Progress,
                           retries: int = 3) -> str:
    """Download bytes[start-end] directly to a temp file. Returns tmp_path."""
    headers = {**DOWNLOAD_HEADERS, "Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        this_attempt = 0
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=READ_SIZE):
                    if chunk:
                        f.write(chunk)
                        progress.add(len(chunk))
                        this_attempt += len(chunk)
            return tmp_path
        except Exception as e:
            # Subtract bytes already counted for this failed attempt
            progress.add(-this_attempt)
            if attempt == retries - 1:
                raise RuntimeError(f"Chunk {start}-{end} failed: {e}")
            time.sleep(2 ** attempt)
    return tmp_path


# ── Assemble temp files ───────────────────────────────────────────────────────

def assemble(tmp_files: List[str], output: str):
    print(f"  Assembling {len(tmp_files)} parts...")
    with open(output, "wb") as out:
        for path in tmp_files:
            with open(path, "rb") as f:
                while True:
                    buf = f.read(4 * 1024 * 1024)  # 4 MB copy buffer
                    if not buf:
                        break
                    out.write(buf)
            os.remove(path)


# ── Main download logic ───────────────────────────────────────────────────────

def fast_download(cdn_url: str, output_path: str,
                  original_url: str = "",
                  threads: int = DEFAULT_THREADS,
                  timeout: int = 60,
                  retries: int = 3,
                  debug: bool = False) -> str:

    filename = output_path or get_filename(cdn_url, original_url=original_url)

    print(f"  Probing server...")
    total, accepts_ranges = probe(cdn_url, debug=debug)

    size_str = f"{total/1024/1024:.1f} MB" if total else "unknown size"
    print(f"  File: {filename}  |  Size: {size_str}  |  Range support: {accepts_ranges}")

    progress = Progress(total)
    tmp_dir = tempfile.mkdtemp(prefix="fileditch_")

    try:
        if accepts_ranges and total > 0 and threads > 1:
            # ── Parallel ──────────────────────────────────────────────────
            min_chunk = 256 * 1024 * 1024  # 256 MB — fewer, larger sequential chunks
            part_size = max(total // threads, min_chunk)
            ranges = []
            start = 0
            while start < total:
                end = min(start + part_size - 1, total - 1)
                ranges.append((start, end))
                start = end + 1

            n = len(ranges)
            print(f"  {n} chunks × {part_size/1024/1024:.1f} MB  |  {threads} threads\n")

            tmp_files = [os.path.join(tmp_dir, f"part_{i:04d}") for i in range(n)]
            failed = False

            with ThreadPoolExecutor(max_workers=threads) as ex:
                futures = {
                    ex.submit(download_chunk_to_file, cdn_url, s, e,
                              tmp_files[i], progress, retries): i
                    for i, (s, e) in enumerate(ranges)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        fut.result()
                    except Exception as err:
                        print(f"\n  [!] Chunk {idx} failed: {err}")
                        failed = True

            progress.done()

            if failed:
                print("[!] Some chunks failed. Try again or reduce --threads.")
                return ""

            assemble(tmp_files, filename)

        else:
            # ── Single-thread fallback ────────────────────────────────────
            reason = "no Range support" if not accepts_ranges else "unknown size"
            print(f"  Single-thread streaming ({reason})...\n")
            with requests.get(cdn_url, headers=DOWNLOAD_HEADERS,
                              stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=READ_SIZE):
                        if chunk:
                            f.write(chunk)
                            progress.add(len(chunk))
            progress.done()
    finally:
        # Clean up temp dir even on error
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    saved = os.path.getsize(filename) if os.path.exists(filename) else 0
    if saved == 0:
        print("[!] Output file is empty.")
        return ""

    print(f"[OK] Saved -> {filename}  ({saved/1024/1024:.2f} MB)")
    return filename


# ── Public entry points ───────────────────────────────────────────────────────

def resolve_output_path(output_dir: "str | None", filename: str) -> str:
    """If output_dir given, create it and join with filename. Else use cwd."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    return filename

def download_file(url: str, output_dir: Optional[str] = None,
                  threads: int = DEFAULT_THREADS,
                  timeout: int = 60,
                  retries: int = 3,
                  debug: bool = False) -> str:

    parsed = urlparse(url)

    if "file.php" not in parsed.path:
        # Direct CDN link
        fname = resolve_output_path(output_dir, get_filename(url))
        return fast_download(url, fname, threads=threads,
                             timeout=timeout, retries=retries, debug=debug)

    cdn_url = resolve_cdn_url(url, debug=debug)
    if not cdn_url:
        return ""

    fname = resolve_output_path(output_dir, get_filename(cdn_url, original_url=url))
    return fast_download(cdn_url, fname, original_url=url,
                         threads=threads, timeout=timeout,
                         retries=retries, debug=debug)


def batch_download(file_path: str, output_dir: str = ".",
                   threads: int = DEFAULT_THREADS,
                   debug: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    urls = [l.strip() for l in open(file_path)
            if l.strip() and not l.strip().startswith("#")]

    print(f"Found {len(urls)} URL(s)\n")
    success = failed = 0
    for i, url in enumerate(urls, 1):
        print(f"\n--- [{i}/{len(urls)}] ---")
        result = download_file(url, output_dir=output_dir, threads=threads, debug=debug)
        (success if result else failed).__class__  # dummy
        if result:
            success += 1
        else:
            failed += 1

    print(f"\nDone!  {success} succeeded   {failed} failed")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel downloader for fileditchfiles.me",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="URL or .txt file (--batch)")
    parser.add_argument("-o", "--output", metavar="DIR", help="Directory to save file into (created if missing)")
    parser.add_argument("-d", "--dir", default=".", help="Output dir (batch mode)")
    parser.add_argument("-t", "--threads", type=int, default=DEFAULT_THREADS,
                        help=f"Parallel threads (default: {DEFAULT_THREADS})")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.batch:
        if not os.path.isfile(args.input):
            print(f"[!] File not found: {args.input}")
            sys.exit(1)
        batch_download(args.input, output_dir=args.dir,
                       threads=args.threads, debug=args.debug)
    else:
        download_file(args.input, output_dir=args.output,
                      threads=args.threads, timeout=args.timeout,
                      retries=args.retries, debug=args.debug)


if __name__ == "__main__":
    main()