#!/usr/bin/env python3
"""
Google Cluster 2019 – Demand-Only Processing Pipeline
=====================================================

Fixes in this version:
----------------------
- Correct timestamp interpretation (microseconds since trace start)
- bucket_s is "seconds since trace start" (NOT a real timestamp)
- Added bucket_index (0-based sequential index)
- Added bucket_dt for VISUALIZATION ONLY (anchored to May 1, 2019)
- Output sorted by bucket_s then machine_id

IMPORTANT: bucket_dt is synthetic and for plotting convenience only.
           All analysis should use bucket_index or bucket_s.
"""

import warnings
import sys
import os
import gzip
import argparse
from glob import glob
from datetime import datetime, timezone
import polars as pl
from tqdm import tqdm

# ============================================================
# macOS warnings fix
# ============================================================

if sys.platform == "darwin":
    warnings.filterwarnings(
        "ignore",
        message="resource_tracker: There appear to be",
        category=UserWarning,
    )


# ============================================================
# Helpers
# ============================================================

def is_valid_gz(path: str) -> bool:
    """Reject empty/corrupted gzip files."""
    try:
        if os.path.getsize(path) < 200:
            return False
    except OSError:
        return False

    try:
        with gzip.open(path, "rb") as f:
            f.read(1)
        return True
    except Exception:
        return False


def list_valid_files(folder: str) -> list[str]:
    paths = sorted(glob(os.path.join(folder, "*.json.gz")))
    return [p for p in paths if is_valid_gz(p)]


# ============================================================
# Parsers
# ============================================================

def parse_usage_file(path: str, max_lines=None):
    lf = pl.scan_ndjson(path)

    schema = lf.collect_schema()
    names = set(schema.names())

    # Extract cpu_rate & mem_usage
    if "average_usage" in names:
        lf = lf.with_columns(
            pl.col("average_usage").struct.field("cpus")
                .cast(pl.Float64).alias("cpu_rate"),
            pl.col("average_usage").struct.field("memory")
                .cast(pl.Float64).alias("mem_usage"),
        )
    else:
        lf = lf.with_columns(
            pl.lit(None, pl.Float64).alias("cpu_rate"),
            pl.lit(None, pl.Float64).alias("mem_usage"),
        )

    lf = lf.select(
        pl.col("start_time").cast(pl.Float64),
        pl.col("end_time").cast(pl.Float64),
        pl.col("machine_id").cast(pl.Int64),
        pl.col("collection_id").cast(pl.Int64),
        pl.col("instance_index").cast(pl.Int64),
        pl.col("cpu_rate"),
        pl.col("mem_usage"),
    )

    if max_lines:
        lf = lf.slice(0, max_lines)

    return lf.collect()


def parse_instance_events_file(path: str, max_lines=None):
    lf = pl.scan_ndjson(path)

    schema = lf.collect_schema()
    names = set(schema.names())

    if "resource_request" in names:
        lf = lf.with_columns(
            pl.col("resource_request").struct.field("cpus")
                .cast(pl.Float64).alias("req_cpus"),
            pl.col("resource_request").struct.field("memory")
                .cast(pl.Float64).alias("req_memory"),
        )
    else:
        lf = lf.with_columns(
            pl.lit(None, pl.Float64).alias("req_cpus"),
            pl.lit(None, pl.Float64).alias("req_memory"),
        )

    if "machine_id" not in names:
        lf = lf.with_columns(pl.lit(None, pl.Int64).alias("machine_id"))

    lf = lf.select(
        pl.col("time").cast(pl.Float64),
        pl.col("type").cast(pl.Int64),
        pl.col("machine_id").cast(pl.Int64),
        pl.col("collection_id").cast(pl.Int64),
        pl.col("instance_index").cast(pl.Int64),
        pl.col("req_cpus"),
        pl.col("req_memory"),
    )

    if max_lines:
        lf = lf.slice(0, max_lines)

    return lf.collect()


def parse_machine_events_file(path: str, max_lines=None):
    lf = pl.scan_ndjson(path)

    lf = lf.select(
        pl.col("time").cast(pl.Float64),
        pl.col("type").cast(pl.Int64),
        pl.col("machine_id").cast(pl.Int64),
    )

    if max_lines:
        lf = lf.slice(0, max_lines)

    return lf.collect()


# ============================================================
# Loader
# ============================================================

def load_sequential(folder, parser_fn, max_files=None, max_lines=None, desc="Loading"):
    files = list_valid_files(folder)
    if max_files:
        files = files[:max_files]

    print(f"📂 {desc}: {len(files)} files")
    dfs = []

    for fp in tqdm(files, desc=desc, unit="file"):
        df = parser_fn(fp, max_lines)
        if df.height > 0:
            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid data loaded from {folder}")

    return pl.concat(dfs, how="diagonal")


# ============================================================
# Aggregations (Demand Only)
# ============================================================

def build_machine_level(usage: pl.DataFrame, events: pl.DataFrame):
    """
    Compute per-machine demand per bucket (NO CAPACITY).
    Also adds:
    - bucket_s (seconds since trace start)
    - bucket_index (0-based sequential index)
    - bucket_dt (synthetic datetime for visualization, anchored to May 1, 2019)
    
    NOTE: bucket_dt is SYNTHETIC and for visualization only.
          The trace does NOT contain real timestamps.
    """

    print("🧮 Building machine-level demand dataset...")

    usage = usage.with_columns([
        ((pl.col("start_time") + pl.col("end_time")) / 2 / 1e6).alias("mid_s"),
    ]).with_columns([
        (pl.col("mid_s") // 300 * 300).alias("bucket_s"),
    ])

    agg = (
        usage.group_by(["bucket_s", "machine_id"])
        .agg([
            pl.sum("cpu_rate").alias("cpu_used"),
            pl.sum("mem_usage").alias("mem_used"),
            pl.len().alias("num_records"),
        ])
        .sort(["bucket_s", "machine_id"])
    )

    # Add bucket_index (0-based sequential index)
    agg = agg.with_columns([
        ((pl.col("bucket_s") / 300) - 1).cast(pl.Int64).alias("bucket_index")
    ])

    # Add synthetic datetime for visualization (anchored to May 1, 2019 00:00:00 UTC)
    # This is NOT a real timestamp - it's for plotting convenience only
    may_2019_epoch = int(datetime(2019, 5, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    agg = agg.with_columns([
        pl.from_epoch(pl.col("bucket_s") + may_2019_epoch, time_unit="s").alias("bucket_dt")
    ])

    # Count new instance arrivals per machine per bucket
    ev = events.filter(pl.col("machine_id").is_not_null()).with_columns(
        (pl.col("time") / 1e6 // 300 * 300).alias("bucket_s")
    )
    evc = ev.group_by(["bucket_s", "machine_id"]).agg(
        pl.len().alias("new_instances_machine")
    )

    ml = agg.join(evc, on=["bucket_s", "machine_id"], how="left")
    ml = ml.with_columns(pl.col("new_instances_machine").fill_null(0))

    return ml


def build_cluster_level(machine: pl.DataFrame, events: pl.DataFrame):
    """
    Cluster-level aggregation, sorted by time.
    Adds:
    - bucket_index (0-based sequential index)
    - bucket_dt (synthetic datetime for visualization, anchored to May 1, 2019)
    
    NOTE: bucket_dt is SYNTHETIC and for visualization only.
          The trace does NOT contain real timestamps.
    """

    print("🧮 Building cluster-level demand dataset...")

    cluster = (
        machine.group_by("bucket_s")
        .agg([
            pl.sum("cpu_used").alias("cpu_demand"),
            pl.sum("mem_used").alias("mem_demand"),
            pl.n_unique("machine_id").alias("machines"),
        ])
        .sort("bucket_s")
    )

    # Add bucket_index (0-based sequential index)
    cluster = cluster.with_columns([
        ((pl.col("bucket_s") / 300) - 1).cast(pl.Int64).alias("bucket_index")
    ])

    # Add synthetic datetime for visualization (anchored to May 1, 2019 00:00:00 UTC)
    # This is NOT a real timestamp - it's for plotting convenience only
    may_2019_epoch = int(datetime(2019, 5, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    cluster = cluster.with_columns([
        pl.from_epoch(pl.col("bucket_s") + may_2019_epoch, time_unit="s").alias("bucket_dt")
    ])

    # Count new instances globally
    ev = events.with_columns(
        (pl.col("time") / 1e6 // 300 * 300).alias("bucket_s")
    )
    evc = ev.group_by("bucket_s").agg(pl.len().alias("new_instances_cluster"))

    cluster = cluster.join(evc, on="bucket_s", how="left")
    cluster = cluster.with_columns(pl.col("new_instances_cluster").fill_null(0))

    return cluster


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--max_files_usage", type=int, default=None)
    parser.add_argument("--max_files_events", type=int, default=None)
    parser.add_argument("--max_files_machines", type=int, default=None)
    parser.add_argument("--max_lines_per_file", type=int, default=None)
    args = parser.parse_args()

    usage = load_sequential(
        os.path.join(args.raw_dir, "instance_usage"),
        parse_usage_file,
        args.max_files_usage,
        args.max_lines_per_file,
        desc="instance_usage"
    )

    events = load_sequential(
        os.path.join(args.raw_dir, "instance_events"),
        parse_instance_events_file,
        args.max_files_events,
        args.max_lines_per_file,
        desc="instance_events"
    )

    machine_events = load_sequential(
        os.path.join(args.raw_dir, "machine_events"),
        parse_machine_events_file,
        args.max_files_machines,
        args.max_lines_per_file,
        desc="machine_events"
    )

    machine_level = build_machine_level(usage, events)
    cluster_level = build_cluster_level(machine_level, events)

    os.makedirs(args.out_dir, exist_ok=True)
    machine_level.write_parquet(os.path.join(args.out_dir, "machine_level.parquet"))
    cluster_level.write_parquet(os.path.join(args.out_dir, "cluster_level.parquet"))

    print("\n✅ DONE — Demand-only processing finished.")
    print(f"   bucket_dt is anchored to May 1, 2019 (for visualization only)")
    print(f"   Use bucket_index or bucket_s for all temporal analysis.\n")


if __name__ == "__main__":
    main()
