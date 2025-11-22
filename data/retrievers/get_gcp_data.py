#!/usr/bin/env python3
import os
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/clusterdata_2019_{cell}/{table}-{index:012d}.json.gz"


# -------------------------------------------------------------
# Download a single shard (silent 404 handling)
# -------------------------------------------------------------
def download_single(url, out_path, progress_desc=""):
    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0) or 0)
            block_size = 1024 * 64

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=progress_desc,
                leave=False,
            ) as pbar:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    f.write(block)
                    pbar.update(len(block))
        return "ok"

    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Silent "end of table" indicator
            if os.path.exists(out_path):
                os.remove(out_path)
            return "not_found"
        else:
            print(f"❌ HTTP error for {url}: {e}")

    except Exception as e:
        print(f"❌ ERROR downloading {url}: {e}")

    if os.path.exists(out_path):
        os.remove(out_path)
    return "error"


# -------------------------------------------------------------
# Find all previously downloaded shards (resume)
# -------------------------------------------------------------
def get_existing_indices(out_dir, table):
    out_dir = Path(out_dir)
    files = list(out_dir.glob(f"{table}-*.json.gz"))

    indices = []
    for f in files:
        try:
            name = f.name  # e.g. instance_usage-000000000031.json.gz
            if name.endswith(".json.gz"):
                name = name[:-8]
            idx = int(name.split("-")[-1])
            indices.append(idx)
        except Exception:
            continue

    return sorted(indices)


# -------------------------------------------------------------
# Download table in parallel with resume + 404 stop
# -------------------------------------------------------------
def download_table(cell, table, end_idx, out_dir, max_workers):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Table: {table}")

    existing = get_existing_indices(out_path, table)

    if existing:
        last = max(existing)
        print(f"  🔍 Found existing indices up to {last} (count={len(existing)})")
        start_idx = last + 1
        print(f"  ⏭️  Resuming from index {start_idx}")
    else:
        start_idx = 0
        print("  🚀 No existing files. Starting from 0")

    if start_idx > end_idx:
        print("  ✔ All requested indices already downloaded.")
        return

    print(f"  ⏳ Downloading shards in range [{start_idx}, {end_idx}] (max_workers={max_workers})")

    MAX_CONSECUTIVE_NOT_FOUND = 3
    consecutive_not_found = 0
    last_not_found_idx = None

    idx = start_idx

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while idx <= end_idx:
            batch_end = min(idx + max_workers - 1, end_idx)
            batch_indices = range(idx, batch_end + 1)

            futures = {
                executor.submit(
                    download_single,
                    BASE_URL.format(cell=cell, table=table, index=i),
                    out_path / f"{table}-{i:012d}.json.gz",
                    f"{table}-{i}",
                ): i
                for i in batch_indices
            }

            try:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Parallel tasks for {table}",
                    unit="file",
                ):
                    i = futures[future]
                    status = future.result()

                    if status == "not_found":
                        consecutive_not_found += 1
                        last_not_found_idx = i
                    else:
                        consecutive_not_found = 0

            except KeyboardInterrupt:
                print("\n🛑 CTRL+C received. Shutting down threads…")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

            if consecutive_not_found >= MAX_CONSECUTIVE_NOT_FOUND:
                print(f"  🛑 Hit {consecutive_not_found} consecutive 404s (last at index {last_not_found_idx}). Assuming end of table.")
                break

            idx = batch_end + 1

    print(f"🎉 Finished table: {table}\n")


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    # Our script lives in: Cloud-AutoScale/data/retrievers/
    # Project root is 2 levels up
    script_root = Path(__file__).resolve().parents[2]
    default_raw = script_root / "data" / "raw"

    parser = argparse.ArgumentParser(description="Parallel GCP 2019 Downloader with Resume & Progress Bars")

    parser.add_argument("--cell", type=str, default="a")

    parser.add_argument(
        "--tables",
        nargs="+",
        default=["instance_usage", "instance_events", "machine_events"],
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=10,
        help="Highest shard index to attempt (inclusive).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,  # uv-safe
        help="Where to store downloaded shards (default: repo_root/data/raw)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Parallel threads for downloading",
    )

    args = parser.parse_args()

    # uv likes to inject "--output_dir .", so we detect and fix
    if args.output_dir is None or args.output_dir.strip() in {"", ".", "./"}:
        args.output_dir = str(default_raw)

    args.output_dir = str(Path(args.output_dir).resolve())

    print(f"📁 Using output_dir = {args.output_dir}")

    for table in args.tables:
        out_dir = Path(args.output_dir) / table
        download_table(args.cell, table, args.end_idx, out_dir, args.threads)
