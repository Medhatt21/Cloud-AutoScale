#!/usr/bin/env python3
"""
Production-grade GCP 2019 processing pipeline.

Features:
- Parallel parsing of .json.gz files using Polars NDJSON lazy scanner
- GZIP corruption detection
- Polars fast processing (vectorized + memory-efficient)
- Accurate midpoint bucketing
- Machine REMOVE event handling
- Progress bars for: loading, parsing, aggregation
- No absolute paths (portable across machines)

OUTPUT:
    data/processed/machine_level.parquet
    data/processed/cluster_level.parquet
"""
import warnings
import sys

# macOS-only: silence false multiprocessing semaphore warnings
if sys.platform == "darwin":
    warnings.filterwarnings(
        "ignore",
        message="resource_tracker: There appear to be",
        category=UserWarning,
    )
import os
import gzip
import argparse
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm


# ============================================================
# Helpers
# ============================================================

def is_valid_gz(path: str) -> bool:
    """Check if .json.gz is readable and non-trivial."""
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
    """Return sorted list of valid (non-corrupted) gzip files."""
    paths = sorted(glob(os.path.join(folder, "*.json.gz")))
    valid = []
    for p in paths:
        if is_valid_gz(p):
            valid.append(p)
        else:
            print(f"⚠️ Skipping corrupted file: {p}")
    return valid


# ============================================================
# Polars-friendly file parsers (lazy NDJSON)
# ============================================================

def parse_usage_file(path: str, max_lines: int | None):
    """
    Parse instance_usage NDJSON.gz into a Polars DataFrame.

    Google 2019 v3 schema: average_usage is a struct:
      average_usage: { cpus: float, memory: float }

    We expose:
      start_time, end_time, machine_id, collection_id, instance_index,
      cpu_rate (from average_usage.cpus),
      mem_usage (from average_usage.memory)
    """
    lf = pl.scan_ndjson(path)

    # Avoid PerformanceWarning: use collect_schema() instead of lf.columns
    schema = lf.collect_schema()
    names = set(schema.names())

    has_avg = "average_usage" in names

    if has_avg:
        lf = lf.with_columns(
            pl.col("average_usage").struct.field("cpus")
            .cast(pl.Float64)
            .alias("cpu_rate"),

            pl.col("average_usage").struct.field("memory")
            .cast(pl.Float64)
            .alias("mem_usage"),
        )
    else:
        lf = lf.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("cpu_rate"),
            pl.lit(None, dtype=pl.Float64).alias("mem_usage"),
        )

    lf = lf.select(
        [
            pl.col("start_time").cast(pl.Float64).alias("start_time"),
            pl.col("end_time").cast(pl.Float64).alias("end_time"),
            pl.col("machine_id").cast(pl.Int64).alias("machine_id"),
            pl.col("collection_id").cast(pl.Int64).alias("collection_id"),
            pl.col("instance_index").cast(pl.Int64).alias("instance_index"),
            pl.col("cpu_rate"),
            pl.col("mem_usage"),
        ]
    )

    if max_lines:
        lf = lf.slice(0, max_lines)

    df = lf.collect()
    return df


def parse_instance_events_file(path: str, max_lines: int | None):
    """
    Parse instance_events NDJSON.gz into a Polars DataFrame.

    Expected fields:
      time, type, machine_id, collection_id, instance_index,
      resource_request: { cpus: float, memory: float }
    """
    lf = pl.scan_ndjson(path)

    schema = lf.collect_schema()
    names = set(schema.names())
    has_rr = "resource_request" in names

    if has_rr:
        lf = lf.with_columns(
            pl.col("resource_request").struct.field("cpus")
            .cast(pl.Float64)
            .alias("req_cpus"),
            pl.col("resource_request").struct.field("memory")
            .cast(pl.Float64)
            .alias("req_memory"),
        )
    else:
        lf = lf.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("req_cpus"),
            pl.lit(None, dtype=pl.Float64).alias("req_memory"),
        )

    lf = lf.select(
        [
            pl.col("time").cast(pl.Float64).alias("time"),
            pl.col("type").cast(pl.Int64).alias("type"),
            pl.col("machine_id").cast(pl.Int64).alias("machine_id"),
            pl.col("collection_id").cast(pl.Int64).alias("collection_id"),
            pl.col("instance_index").cast(pl.Int64).alias("instance_index"),
            pl.col("req_cpus"),
            pl.col("req_memory"),
        ]
    )

    if max_lines:
        lf = lf.slice(0, max_lines)

    df = lf.collect()
    return df


def parse_machine_events_file(path: str, max_lines: int | None):
    """
    Parse machine_events NDJSON.gz into a Polars DataFrame.

    Expected fields:
      time, type, machine_id, cpu_capacity, memory_capacity, platform_id, clock_rate
    """
    lf = pl.scan_ndjson(path)

    lf = lf.select(
        [
            pl.col("time").cast(pl.Float64).alias("time"),
            pl.col("type").cast(pl.Int64).alias("type"),
            pl.col("machine_id").cast(pl.Int64).alias("machine_id"),
            pl.col("cpu_capacity").cast(pl.Float64).alias("cpu_capacity"),
            pl.col("memory_capacity").cast(pl.Float64).alias("memory_capacity"),
            pl.col("platform_id").cast(pl.Int64).alias("platform_id"),
            pl.col("clock_rate").cast(pl.Float64).alias("clock_rate"),
        ]
    )

    if max_lines:
        lf = lf.slice(0, max_lines)

    df = lf.collect()
    return df


# ============================================================
# Master parallel loader
# ============================================================

def load_parallel(folder: str, parser_fn, max_files=None, max_lines=None, desc="Loading"):
    files = list_valid_files(folder)
    if not files:
        print(f"⚠️ No valid files found in {folder}")
        return None

    if max_files:
        files = files[:max_files]

    print(f"📂 {desc}: {len(files)} files from {folder}")

    results = []

    # Avoid oversubscribing the CPU: cap at 8 workers by default
    max_workers = min(os.cpu_count() or 4, len(files), 8)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(parser_fn, path, max_lines) for path in files]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
            unit="file",
        ):
            df = fut.result()
            if df is not None and df.height > 0:
                results.append(df)

    if not results:
        return None

    return pl.concat(results, how="diagonal")


# ============================================================
# Aggregations
# ============================================================

def summarize_machine_capacity(machine_events: pl.DataFrame) -> pl.DataFrame:
    print("🧮 Summarizing machine capacities...")

    me = machine_events.with_columns(
        [
            pl.when(pl.col("type") == 1)  # 1 == REMOVE in GCP traces
            .then(0)
            .otherwise(pl.col("cpu_capacity"))
            .alias("cpu_capacity_clean"),

            pl.when(pl.col("type") == 1)
            .then(0)
            .otherwise(pl.col("memory_capacity"))
            .alias("memory_capacity_clean"),
        ]
    )

    caps = (
        me.group_by("machine_id")
        .agg(
            [
                pl.max("cpu_capacity_clean").alias("cpu_capacity"),
                pl.max("memory_capacity_clean").alias("memory_capacity"),
                pl.max("platform_id").alias("platform_id"),
                pl.max("clock_rate").alias("clock_rate"),
            ]
        )
    )

    print(f"✔ Machine types detected: {caps.height}")
    return caps


def build_machine_level(usage: pl.DataFrame, caps: pl.DataFrame, events: pl.DataFrame | None):
    print("🧮 Building machine-level dataset...")

    # Midpoint time in seconds, bucketed into 5-min (300s) windows
    usage = usage.with_columns(
        ((pl.col("start_time") + pl.col("end_time")) / 2 / 1e6).alias("mid_s")
    ).with_columns(
        (pl.col("mid_s") // 300 * 300).alias("bucket_s")
    )

    usage_agg = (
        usage.group_by(["bucket_s", "machine_id"])
        .agg(
            [
                pl.sum("cpu_rate").alias("cpu_used"),
                pl.sum("mem_usage").alias("mem_used"),
                pl.count().alias("num_records"),
                pl.n_unique("collection_id").alias("unique_jobs"),
                pl.n_unique("instance_index").alias("unique_instances"),
            ]
        )
        .sort(["bucket_s", "machine_id"])
    )

    # Join with capacities
    machine_level = usage_agg.join(caps, on="machine_id", how="left")

    # Per-machine instance events (starts) per bucket
    if events is not None and "time" in events.columns:
        ev = events.with_columns(
            (pl.col("time") / 1e6 // 300 * 300).alias("bucket_s")
        )
        ev_agg = (
            ev.group_by(["bucket_s", "machine_id"])
            .agg(pl.count().alias("new_instances_machine"))
        )
        machine_level = machine_level.join(ev_agg, on=["bucket_s", "machine_id"], how="left")
        machine_level = machine_level.with_columns(
            pl.col("new_instances_machine").fill_null(0)
        )
    else:
        machine_level = machine_level.with_columns(
            pl.lit(0).alias("new_instances_machine")
        )

    # Utilization
    machine_level = machine_level.with_columns(
        [
            (pl.col("cpu_used") / pl.col("cpu_capacity")).alias("utilization_cpu"),
            (pl.col("mem_used") / pl.col("memory_capacity")).alias("utilization_mem"),
        ]
    )

    return machine_level


def build_cluster_level(machine: pl.DataFrame, events: pl.DataFrame | None):
    print("🧮 Building cluster-level dataset...")

    cluster = (
        machine.group_by("bucket_s")
        .agg(
            [
                pl.sum("cpu_used").alias("cpu_demand"),
                pl.sum("mem_used").alias("mem_demand"),
                pl.sum("cpu_capacity").alias("cpu_capacity"),
                pl.sum("memory_capacity").alias("mem_capacity"),
                pl.n_unique("machine_id").alias("machines"),
                pl.mean("utilization_cpu").alias("avg_utilization_cpu"),
                pl.mean("utilization_mem").alias("avg_utilization_mem"),
            ]
        )
        .sort("bucket_s")
    )

    if events is not None and "time" in events.columns:
        ev = events.with_columns(
            (pl.col("time") / 1e6 // 300 * 300).alias("bucket_s")
        )
        evc = ev.group_by("bucket_s").agg(pl.count().alias("new_instances_cluster"))
        cluster = cluster.join(evc, on="bucket_s", how="left")
        cluster = cluster.with_columns(
            pl.col("new_instances_cluster").fill_null(0)
        )
    else:
        cluster = cluster.with_columns(
            pl.lit(0).alias("new_instances_cluster")
        )

    return cluster


# ============================================================
# Main
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

    usage_dir = os.path.join(args.raw_dir, "instance_usage")
    events_dir = os.path.join(args.raw_dir, "instance_events")
    machines_dir = os.path.join(args.raw_dir, "machine_events")

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== 🚀 Loading Google Traces in Parallel ===")

    instance_usage = load_parallel(
        usage_dir,
        parse_usage_file,
        max_files=args.max_files_usage,
        max_lines=args.max_lines_per_file,
        desc="instance_usage",
    )

    instance_events = load_parallel(
        events_dir,
        parse_instance_events_file,
        max_files=args.max_files_events,
        max_lines=args.max_lines_per_file,
        desc="instance_events",
    )

    machine_events = load_parallel(
        machines_dir,
        parse_machine_events_file,
        max_files=args.max_files_machines,
        max_lines=args.max_lines_per_file,
        desc="machine_events",
    )

    if instance_usage is None or machine_events is None:
        raise RuntimeError("Missing required data (usage or machine_events). Check raw_dir layout.")

    # Summaries
    caps = summarize_machine_capacity(machine_events)

    # Machine-level
    machine_level = build_machine_level(instance_usage, caps, instance_events)
    machine_out = os.path.join(args.out_dir, "machine_level.parquet")
    print(f"💾 Saving machine-level → {machine_out}")
    machine_level.write_parquet(machine_out)

    # Cluster-level
    cluster_level = build_cluster_level(machine_level, instance_events)
    cluster_out = os.path.join(args.out_dir, "cluster_level.parquet")
    print(f"💾 Saving cluster-level → {cluster_out}")
    cluster_level.write_parquet(cluster_out)

    print("\n🎉 DONE — Production-grade processing completed successfully")


if __name__ == "__main__":
    main()
