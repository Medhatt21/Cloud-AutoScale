#!/usr/bin/env python3
import os
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/clusterdata_2019_{cell}/{table}-{index:012d}.json.gz"


# -------------------------------------------------------------
# Helper: safe download with overwrite + integrity cleanup
# -------------------------------------------------------------
def download_single(url, out_path, progress_desc=""):
    try:
        # Stream download with progress bar
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 64  # 64KB chunks

            with open(out_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=progress_desc,
                leave=False
            ) as pbar:
                for block in iter(lambda: response.read(block_size), b""):
                    f.write(block)
                    pbar.update(len(block))

        return True

    except Exception as e:
        print(f"❌ ERROR downloading {url}: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


# -------------------------------------------------------------
# Determine existing files
# -------------------------------------------------------------
def get_existing_indices(out_dir, table):
    files = list(Path(out_dir).glob(f"{table}-*.json.gz"))
    indices = []
    for f in files:
        try:
            idx = int(f.stem.split("-")[-1])
            indices.append(idx)
        except:
            continue
    return sorted(indices)


# -------------------------------------------------------------
# Main download logic (parallel)
# -------------------------------------------------------------
def download_table(cell, table, end_idx, out_dir, max_workers):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📦 Table: {table}")

    existing = get_existing_indices(out_dir, table)

    if existing:
        print(f"  🔍 Found existing: {existing[:5]} ... total={len(existing)}")
        start_idx = max(existing)  # re-download last
        print(f"  ♻️ Re-downloading last existing index {start_idx} to ensure integrity")
    else:
        start_idx = 0
        print("  🚀 No existing files. Starting from 0")

    if end_idx < start_idx:
        print("  ✔ All files already downloaded.")
        return

    # List of (index, url, path)
    tasks = []
    for idx in range(start_idx, end_idx + 1):
        url = BASE_URL.format(cell=cell, table=table, index=idx)
        out_path = os.path.join(out_dir, f"{table}-{idx:012d}.json.gz")
        tasks.append((idx, url, out_path))

    print(f"  ⏳ Downloading {len(tasks)} files in parallel (max_workers={max_workers})")

    # ---------------------------------------------------------
    # Execute downloads in parallel with ThreadPoolExecutor
    # ---------------------------------------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_single,
                url,
                out_path,
                progress_desc=f"{table}-{idx}"
            ): idx
            for idx, url, out_path in tasks
        }

        # Wait for completion
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Parallel tasks for {table}", unit="file"):
            idx = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"  ⚠️ Failed downloading file index {idx}")
            except Exception as e:
                print(f"  ❌ Unhandled exception for index {idx}: {e}")

    print(f"🎉 Finished table: {table}\n")


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel GCP 2019 Downloader with Resume & Progress Bars")
    parser.add_argument("--cell", type=str, default="a")
    parser.add_argument("--tables", nargs="+",
                        default=["instance_usage", "instance_events", "machine_events"])

    parser.add_argument("--end_idx", type=int, default=10,
                        help="Default: first 10 shards only")

    parser.add_argument("--output_dir", type=str, default="./data/raw")

    parser.add_argument("--threads", type=int, default=4,
                        help="Parallel threads for downloading")

    args = parser.parse_args()

    for table in args.tables:
        out_dir = os.path.join(args.output_dir, table)
        download_table(args.cell, table, args.end_idx, out_dir, args.threads)
