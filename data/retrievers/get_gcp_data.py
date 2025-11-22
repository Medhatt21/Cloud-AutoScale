#!/usr/bin/env python3
import os
import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/clusterdata_2019_{cell}/{table}-{index:012d}.json.gz"

def download_file(url, out_path):
    """
    Downloads a file and overwrites if exists.
    """
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"  ✅ Downloaded: {out_path}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {url} | Error: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


def get_existing_indices(out_dir, table):
    """
    Detects which shards already exist: table-000000000XXX.json.gz
    Returns a sorted list of integer indices.
    """
    files = list(Path(out_dir).glob(f"{table}-*.json.gz"))
    indices = []
    for f in files:
        try:
            idx = int(f.stem.split("-")[-1])
            indices.append(idx)
        except ValueError:
            continue

    return sorted(indices)


def download_table(cell, table, end_idx, out_dir):
    """
    Improved logic:
    - Checks existing files
    - Resumes automatically
    - Re-downloads last file for safety
    - Default stops at end_idx (10 by default)
    """

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📦 Processing table: {table}")
    existing = get_existing_indices(out_dir, table)

    if existing:
        print(f"  🔍 Found existing files: {existing[:5]} ... total={len(existing)}")

        last_ok = max(existing)

        # Always re-download last file to avoid partial / broken downloads
        start_idx = last_ok
        print(f"  ♻️  Re-downloading last file #{start_idx} to ensure integrity.")
    else:
        start_idx = 0
        print("  🚀 No existing files. Starting fresh.")

    if end_idx < start_idx:
        print(f"  ✔ All required files already downloaded. (end_idx={end_idx})")
        return

    print(f"  ⏳ Resuming from index {start_idx} up to {end_idx}")

    for idx in range(start_idx, end_idx + 1):
        url = BASE_URL.format(cell=cell, table=table, index=idx)
        out_path = os.path.join(out_dir, f"{table}-{idx:012d}.json.gz")

        print(f"    → Downloading {table} [{idx}]")

        if os.path.exists(out_path) and idx != start_idx:
            print(f"      ⚡ Skipping, file exists: {out_path}")
            continue

        success = download_file(url, out_path)
        if not success:
            print("      ⚠️ Warning: download failed, continuing…")


    print(f"\n  🎉 Finished table: {table}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restartable Google Cluster 2019 trace downloader")

    parser.add_argument("--cell", type=str, default="a")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=["instance_usage", "instance_events", "machine_events"]
    )

    parser.add_argument(
        "--end_idx", type=int, default=10,
        help="Default: 10 shards"
    )

    parser.add_argument(
        "--output_dir", type=str, default="./data/raw"
    )

    args = parser.parse_args()

    for table in args.tables:
        out_dir = os.path.join(args.output_dir, table)
        download_table(args.cell, table, args.end_idx, out_dir)
