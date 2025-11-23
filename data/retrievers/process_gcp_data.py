#!/usr/bin/env python3
"""
Production-grade GCP 2019 processing pipeline (final version).

Key Features
------------
- Parses NDJSON.gz using Polars lazy API
- Handles corrupted or partial shards
- Inference of machine CPU & Memory capacity via P99 usage
- Capacity rounded to realistic machine types
- Handles instance_events shards missing machine_id cleanly
- 5-minute bucketing using job midpoint
- Warning-free (no polars deprecation warnings)
"""

import warnings
import sys
import os
import gzip
import argparse
from glob import glob
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

    # Extract average_usage struct if present
    if "average_usage" in names:
        lf = lf.with_columns(
            pl.col("average_usage").struct.field("cpus").cast(pl.Float64).alias("cpu_rate"),
            pl.col("average_usage").struct.field("memory").cast(pl.Float64).alias("mem_usage"),
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

    # Extract resource_request if present
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

    # Ensure machine_id always exists (as null if missing)
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
    """
    Note: machine_events in v3 dataset lacks capacity info entirely.
    Only machine_id and REMOVE events are useful.
    """
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

    for f in tqdm(files, desc=desc, unit="file"):
        try:
            df = parser_fn(f, max_lines)
            if df.height > 0:
                dfs.append(df)
        except Exception as e:
            print(f"❌ Error parsing {f}: {e}")
            raise

    return pl.concat(dfs, how="diagonal")


# ============================================================
# Capacity Inference
# ============================================================

def round_cpu(x: float) -> int:
    if x < 1: return 1
    if x < 2: return 2
    if x < 4: return 4
    if x < 8: return 8
    if x < 16: return 16
    if x < 32: return 32
    return int(round(x))


def round_mem_gb(x_mb: float) -> int:
    gb = x_mb / 1024
    if gb < 1: return 1
    if gb < 4: return 4
    if gb < 8: return 8
    if gb < 16: return 16
    if gb < 32: return 32
    if gb < 64: return 64
    if gb < 128: return 128
    return int(round(gb))


def infer_machine_capacity_from_usage(usage: pl.DataFrame):
    print("🧮 Inferring machine capacity from P99 usage...")

    caps = (
        usage
        .group_by("machine_id")
        .agg([
            pl.quantile("cpu_rate", 0.99).alias("cpu_p99"),
            pl.quantile("mem_usage", 0.99).alias("mem_p99"),
        ])
        .with_columns([
            pl.col("cpu_p99").map_elements(round_cpu).alias("cpu_capacity"),
            pl.col("mem_p99").map_elements(lambda x: round_mem_gb(x) * 1024).alias("memory_capacity"),
        ])
        .select(["machine_id", "cpu_capacity", "memory_capacity"])
    )

    print(f"✔ Inferred capacity for {caps.height} machines")
    return caps


# ============================================================
# Aggregations
# ============================================================

def build_machine_level(usage, caps, events):
    print("🧮 Building machine-level dataset...")

    usage = usage.with_columns(
        ((pl.col("start_time") + pl.col("end_time")) / 2 / 1e6).alias("mid_s")
    ).with_columns(
        (pl.col("mid_s") // 300 * 300).alias("bucket_s")
    )

    agg = (
        usage.group_by(["bucket_s", "machine_id"])
        .agg([
            pl.sum("cpu_rate").alias("cpu_used"),
            pl.sum("mem_usage").alias("mem_used"),
            pl.len().alias("num_records"),
        ])
        .sort(["bucket_s", "machine_id"])
    )

    ml = agg.join(caps, on="machine_id", how="left")

    # Machine-level event counts (only events with machine_id)
    ev = (
        events
        .filter(pl.col("machine_id").is_not_null())
        .with_columns((pl.col("time") / 1e6 // 300 * 300).alias("bucket_s"))
    )
    evc = ev.group_by(["bucket_s", "machine_id"]).agg(
        pl.len().alias("new_instances_machine")
    )

    ml = ml.join(evc, on=["bucket_s", "machine_id"], how="left")
    ml = ml.with_columns(pl.col("new_instances_machine").fill_null(0))

    ml = ml.with_columns([
        (pl.col("cpu_used") / pl.col("cpu_capacity")).alias("util_cpu"),
        (pl.col("mem_used") / pl.col("memory_capacity")).alias("util_mem"),
    ])

    return ml


def build_cluster_level(machine, events):
    print("🧮 Building cluster-level dataset...")

    cluster = (
        machine.group_by("bucket_s")
        .agg([
            pl.sum("cpu_used").alias("cpu_demand"),
            pl.sum("mem_used").alias("mem_demand"),
            pl.sum("cpu_capacity").alias("cpu_capacity"),
            pl.sum("memory_capacity").alias("mem_capacity"),
            pl.n_unique("machine_id").alias("machines"),
            pl.mean("util_cpu").alias("avg_util_cpu"),
            pl.mean("util_mem").alias("avg_util_mem"),
        ])
    )

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

    # Capacity inference from usage
    caps = infer_machine_capacity_from_usage(usage)

    # Build datasets
    machine_level = build_machine_level(usage, caps, events)
    cluster_level = build_cluster_level(machine_level, events)

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    machine_level.write_parquet(os.path.join(args.out_dir, "machine_level.parquet"))
    cluster_level.write_parquet(os.path.join(args.out_dir, "cluster_level.parquet"))

    print("\n🎉 DONE — Processing finished successfully.\n")


if __name__ == "__main__":
    main()
