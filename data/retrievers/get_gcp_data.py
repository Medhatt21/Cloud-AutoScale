import os
import argparse
import urllib.request

BASE_URL = "https://storage.googleapis.com/clusterdata_2019_{cell}/{table}-{index:012d}.json.gz"

def download_file(url, out_path):
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"✅ Downloaded: {out_path}")
    except Exception as e:
        print(f"❌ Failed: {url} | Error: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)

def download_table(cell, table, start_idx, end_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(start_idx, end_idx + 1):
        url = BASE_URL.format(cell=cell, table=table, index=idx)
        out_path = os.path.join(out_dir, f"{table}-{idx:012d}.json.gz")
        print(f"⏳ Downloading {table} [{idx}]")

        download_file(url, out_path)

    print(f"\n📦 Finished downloading {table}: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 1 month of Google Cluster 2019 trace data")
    parser.add_argument("--cell", type=str, default="a")
    parser.add_argument("--tables", nargs="+", default=["instance_usage", "instance_events", "machine_events"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=150, help="~150 shards ≈ 1 month")
    parser.add_argument("--output_dir", type=str, default="./data/raw")

    args = parser.parse_args()

    for table in args.tables:
        out_dir = os.path.join(args.output_dir, table)
        download_table(args.cell, table, args.start_idx, args.end_idx, out_dir)
