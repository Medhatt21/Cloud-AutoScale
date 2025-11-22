#!/usr/bin/env python3
"""
Production-grade GCP 2019 processing pipeline.

Features:
- Parallel parsing of .json.gz files
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

import os
import gzip
import argparse
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import orjson
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


def load_json_gz(path: str, max_lines: int | None = None) -> list[dict]:
    """Load gzipped JSON lines manually using orjson."""
    rows = []
    with gzip.open(path, "rb") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            rows.append(orjson.loads(line))
    return rows


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
# Polars-friendly parallel file loader
# ============================================================

def parse_usage_file(path: str, max_lines: int | None):
    rows = load_json_gz(path, max_lines)
    if not rows:
        return None

    df = pl.DataFrame(
        {
            "start_time": [r.get("start_time") for r in rows],
            "end_time": [r.get("end_time") for r in rows],
            "cpu_rate": [r.get("cpu_rate") for r in rows],
            "mem_usage": [r.get("mem_usage") for r in rows],
            "machine_id": [r.get("machine_id") for r in rows],
            "collection_id": [r.get("collection_id") for r in rows],
            "instance_index": [r.get("instance_index") for r in rows],
        }
    )

    return df


def parse_instance_events_file(path: str, max_lines: int | None):
    rows = load_json_gz(path, max_lines)
    if not rows:
        return None

    df = pl.DataFrame(
        {
            "time": [r.get("time") for r in rows],
            "type": [r.get("type") for r in rows],
            "machine_id": [r.get("machine_id") for r in rows],
            "collection_id": [r.get("collection_id") for r in rows],
            "instance_index": [r.get("instance_index") for r in rows],
            "req_cpus": [(r.get("resource_request") or {}).get("cpus") for r in rows],
            "req_memory": [(r.get("resource_request") or {}).get("memory") for r in rows],
        }
    )
    return df


def parse_machine_events_file(path: str, max_lines: int | None):
    rows = load_json_gz(path, max_lines)
    if not rows:
        return None

    df = pl.DataFrame(
        {
            "time": [r.get("time") for r in rows],
            "type": [r.get("type") for r in rows],
            "machine_id": [r.get("machine_id") for r in rows],
            "cpu_capacity": [r.get("cpu_capacity") for r in rows],
            "memory_capacity": [r.get("memory_capacity") for r in rows],
            "platform_id": [r.get("platform_id") for r in rows],
            "clock_rate": [r.get("clock_rate") for r in rows],
        }
    )
    return df


# ============================================================
# Master parallel loader
# ============================================================

def load_parallel(folder: str, parser_fn, max_files=None, max_lines=None, desc="Loading"):
    files = list_valid_files(folder)
    if max_files:
        files = files[:max_files]

    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = []
        for path in files:
            futures.append(ex.submit(parser_fn, path, max_lines))

        for f in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file"):
            df = f.result()
            if df is not None:
                results.append(df)

    if not results:
        return None

    return pl.concat(results, how="diagonal")


# ============================================================
# Aggregations
# ============================================================

def summarize_machine_capacity(machine_events: pl.DataFrame) -> pl.DataFrame:
    print("🧮 Summarizing machine capacities...")

    # Handle REMOVE events: capacity → 0
    me = machine_events.with_columns(
        [
            pl.when(pl.col("type") == "REMOVE")
            .then(0)
            .otherwise(pl.col("cpu_capacity"))
            .alias("cpu_capacity_clean"),

            pl.when(pl.col("type") == "REMOVE")
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

    # Midpoint time
    usage = usage.with_columns(
        ((pl.col("start_time") + pl.col("end_time")) / 2 / 1e6)
        .alias("mid_s")
    ).with_columns(
        (pl.col("mid_s") // 300 * 300).alias("bucket_s")
    )

    # Aggregate per machine per bucket
    usage_agg = (
        usage.group_by(["bucket_s", "machine_id"])
        .agg(
            [
                pl.sum("cpu_rate").alias("cpu_used"),
                pl.sum("mem_usage").alias("mem_used"),
                pl.count().alias("num_instances"),
                pl.n_unique("collection_id").alias("unique_jobs"),
                pl.n_unique("instance_index").alias("unique_instances"),
            ]
        )
        .sort(["bucket_s", "machine_id"])
    )

    # Join with capacities
    machine_level = usage_agg.join(caps, on="machine_id", how="left")

    # Add per-machine new instance starts
    if events is not None:
        ev_agg = (
            events.group_by(["bucket_s", "machine_id"])
            .agg(pl.count().alias("new_instances_machine"))
        )
        machine_level = machine_level.join(ev_agg, on=["bucket_s", "machine_id"], how="left")
        machine_level = machine_level.with_columns(
            pl.col("new_instances_machine").fill_null(0)
        )
    else:
        machine_level = machine_level.with_columns(pl.lit(0).alias("new_instances_machine"))

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

    if events is not None:
        evc = events.group_by("bucket_s").agg(pl.count().alias("new_instances_cluster"))
        cluster = cluster.join(evc, on="bucket_s", how="left")
        cluster = cluster.with_columns(pl.col("new_instances_cluster").fill_null(0))
    else:
        cluster = cluster.with_columns(pl.lit(0).alias("new_instances_cluster"))

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
        usage_dir, parse_usage_file,
        max_files=args.max_files_usage,
        max_lines=args.max_lines_per_file,
        desc="instance_usage"
    )

    instance_events = load_parallel(
        events_dir, parse_instance_events_file,
        max_files=args.max_files_events,
        max_lines=args.max_lines_per_file,
        desc="instance_events"
    )

    machine_events = load_parallel(
        machines_dir, parse_machine_events_file,
        max_files=args.max_files_machines,
        max_lines=args.max_lines_per_file,
        desc="machine_events"
    )

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
