#!/usr/bin/env python3
"""
scripts/fast_download_fastq.py

High-speed multi-threaded segmented downloader for ENA FASTQ files.
Downloads large FASTQ.gz files using concurrent HTTP Range requests with resume support.
"""

import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_file_size(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "Mozilla/5.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                length = resp.headers.get("Content-Length")
                if length:
                    return int(length)
        except Exception as e:
            print(f"HEAD request attempt {attempt+1} failed for {url}: {e}", flush=True)
            time.sleep(2)
    raise RuntimeError(f"Could not retrieve file size for {url}")

def download_chunk(url, start_byte, end_byte, part_path, max_retries=50):
    total_chunk_len = end_byte - start_byte + 1
    
    for attempt in range(max_retries):
        existing_len = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if existing_len == total_chunk_len:
            return part_path
        elif existing_len > total_chunk_len:
            with open(part_path, "r+b") as f:
                f.truncate(total_chunk_len)
            return part_path
        
        cur_start = start_byte + existing_len
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={cur_start}-{end_byte}")
            req.add_header("User-Agent", "Mozilla/5.0")
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 206:
                    if os.path.exists(part_path):
                        os.remove(part_path)
                    raise RuntimeError(f"Expected HTTP 206 for range, got {resp.status}. Resetting chunk.")
                
                with open(part_path, "ab" if existing_len > 0 else "wb") as f:
                    remaining = total_chunk_len - existing_len
                    while remaining > 0:
                        to_read = min(1024 * 1024, remaining)
                        buf = resp.read(to_read)
                        if not buf:
                            break
                        f.write(buf)
                        remaining -= len(buf)
            
            act_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if act_size == total_chunk_len:
                return part_path
            elif act_size > total_chunk_len:
                with open(part_path, "r+b") as f:
                    f.truncate(total_chunk_len)
                return part_path
        except Exception as e:
            backoff = min(15, 2 + (attempt % 8) * 2)
            print(f"Retry {attempt+1}/{max_retries} on {os.path.basename(part_path)}: {e} (sleeping {backoff}s)", flush=True)
            time.sleep(backoff)

    raise RuntimeError(f"Failed to download chunk {part_path} after {max_retries} attempts")

def download_file_parallel(url, out_path, num_workers=6):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    total_size = get_file_size(url)
    if os.path.exists(out_path) and os.path.getsize(out_path) == total_size:
        import subprocess
        res = subprocess.run(["gzip", "-t", out_path], capture_output=True)
        if res.returncode == 0:
            print(f"File already complete & verified: {out_path} ({total_size / (1024**2):.2f} MB)", flush=True)
            return True
        else:
            print(f"WARNING: {out_path} size matches ({total_size} bytes) but gzip -t failed. Removing corrupt file and re-downloading...", flush=True)
            os.remove(out_path)

    print(f"\nDownloading {os.path.basename(out_path)} ({total_size / (1024**2):.2f} MB) using {num_workers} parallel threads...", flush=True)
    start_time = time.time()
    
    chunk_size = total_size // num_workers
    futures = []
    part_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            start_b = i * chunk_size
            end_b = total_size - 1 if i == num_workers - 1 else (i + 1) * chunk_size - 1
            part_path = f"{out_path}.part_{i}"
            part_files.append(part_path)
            futures.append(executor.submit(download_chunk, url, start_b, end_b, part_path))

        done_cnt = 0
        for f in as_completed(futures):
            f.result()
            done_cnt += 1
            pct = (done_cnt / num_workers) * 100
            print(f"  [{os.path.basename(out_path)}] Completed chunk {done_cnt}/{num_workers} ({pct:.0f}%)", flush=True)

    # Verify every part exists and has the exact expected size before assembling
    for i, p in enumerate(part_files):
        start_b = i * chunk_size
        end_b = total_size - 1 if i == num_workers - 1 else (i + 1) * chunk_size - 1
        expected_p_len = end_b - start_b + 1
        act_len = os.path.getsize(p) if os.path.exists(p) else 0
        if act_len != expected_p_len:
            raise RuntimeError(f"Part {p} size mismatch: got {act_len}, expected {expected_p_len}")

    print(f"Assembling parts for {os.path.basename(out_path)}...", flush=True)
    temp_out = f"{out_path}.assembling"
    with open(temp_out, "wb") as outfile:
        for p in part_files:
            with open(p, "rb") as infile:
                while True:
                    buf = infile.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    outfile.write(buf)
            os.remove(p)

    assembled_size = os.path.getsize(temp_out)
    if assembled_size != total_size:
        os.remove(temp_out)
        raise RuntimeError(f"Assembled file {temp_out} size {assembled_size} != expected {total_size}")

    os.replace(temp_out, out_path)
    import subprocess
    res = subprocess.run(["gzip", "-t", out_path], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: gzip -t failed on {out_path} with exit code {res.returncode}", flush=True)
        print(f"STDERR: {res.stderr.strip()}", flush=True)
        print(f"STDOUT: {res.stdout.strip()}", flush=True)
        raise RuntimeError(f"Downloaded file {out_path} failed gzip -t integrity check: {res.stderr.strip()}")
    elapsed = time.time() - start_time
    speed_mb = (total_size / (1024**2)) / max(elapsed, 0.01)
    print(f"SUCCESS: {os.path.basename(out_path)} finished in {elapsed:.1f}s ({speed_mb:.2f} MB/s)", flush=True)
    return True

def main():
    runs = sys.argv[1:] if len(sys.argv) > 1 else ["ERR127306", "ERR127307"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "raw")
    
    for run in runs:
        prefix = run[:6]
        for pair in [1, 2]:
            url = f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}/{run}/{run}_{pair}.fastq.gz"
            out_file = os.path.join(out_dir, f"{run}_{pair}.fastq.gz")
            download_file_parallel(url, out_file, num_workers=6)

if __name__ == "__main__":
    main()
