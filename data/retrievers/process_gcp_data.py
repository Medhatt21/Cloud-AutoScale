#!/usr/bin/env python
"""
Process Google Cluster Data 2019 (1 month) into:

1) Machine-level buckets (per machine, per 5-minute interval)
2) Cluster-level buckets (sum over machines per 5-minute interval)

Expected raw layout (from your download script):

data/raw/
    instance_usage/
        instance_usage-000000000000.json.gz
        ...
    instance_events/
        instance_events-000000000000.json.gz
        ...
    machine_events/
        machine_events-000000000000.json.gz
        ...

Outputs:

data/processed/machine_level.parquet
data/processed/cluster_level.parquet

You can control how much data to load using:
  --max_files_usage
  --max_files_events
  --max_files_machines
  --max_lines_per_file
"""

import os
import gzip
import argparse
from glob import glob

import orjson
import polars as pl


# -------------------------------------------------------------------
# Low-level loader
# -------------------------------------------------------------------

def load_json_gz(path: str, max_lines: int | None = None) -> list[dict]:
    rows = []
    with gzip.open(path, "rb") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            rows.append(orjson.loads(line))
    return rows


# -------------------------------------------------------------------
# Loaders for each table
# -------------------------------------------------------------------

def load_instance_usage(
    folder: str,
    max_files: int | None = None,
    max_lines_per_file: int | None = None,
) -> pl.DataFrame:
    """
    Load instance_usage shards and return a Polars DataFrame with:

    start_time, end_time (microseconds)
    cpu_rate, mem_usage
    machine_id, collection_id, instance_index
    """
    print(f"📥 Loading instance_usage from {folder}")
    files = sorted(glob(os.path.join(folder, "*.json.gz")))
    if max_files is not None:
        files = files[:max_files]

    df_list: list[pl.DataFrame] = []
    for path in files:
        print(f"  → {os.path.basename(path)}")
        rows = load_json_gz(path, max_lines_per_file)
        if not rows:
            continue

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
        df_list.append(df)

    if not df_list:
        raise RuntimeError("No instance_usage data loaded")

    df = pl.concat(df_list, how="diagonal")

    # Convert microseconds → seconds and bucket into 5-minute windows (300s)
    df = df.with_columns(
        [
            (pl.col("start_time") / 1e6).alias("start_s"),
            (pl.col("end_time") / 1e6).alias("end_s"),
        ]
    ).with_columns(
        [
            (pl.col("start_s") // 300 * 300).alias("bucket_s"),
        ]
    )

    print(f"✅ instance_usage loaded: {df.height} rows")
    return df


def load_instance_events(
    folder: str,
    max_files: int | None = None,
    max_lines_per_file: int | None = None,
) -> pl.DataFrame | None:
    """
    Load instance_events shards and return a Polars DataFrame with:

    time (microseconds)
    type (event type)
    machine_id (may be null for some events)
    collection_id, instance_index
    resource_request.{cpus, memory}

    Used mainly to count "new_instances" per bucket and optionally per machine.
    """
    if not os.path.isdir(folder):
        print(f"⚠️ instance_events folder {folder} not found, skipping.")
        return None

    print(f"📥 Loading instance_events from {folder}")
    files = sorted(glob(os.path.join(folder, "*.json.gz")))
    if not files:
        print("⚠️ No instance_events files found, skipping.")
        return None
    if max_files is not None:
        files = files[:max_files]

    df_list: list[pl.DataFrame] = []
    for path in files:
        print(f"  → {os.path.basename(path)}")
        rows = load_json_gz(path, max_lines_per_file)
        if not rows:
            continue

        df = pl.DataFrame(
            {
                "time": [r.get("time") for r in rows],
                "type": [r.get("type") for r in rows],
                "machine_id": [r.get("machine_id") for r in rows],
                "collection_id": [r.get("collection_id") for r in rows],
                "instance_index": [r.get("instance_index") for r in rows],
                "req_cpus": [
                    (r.get("resource_request") or {}).get("cpus") for r in rows
                ],
                "req_memory": [
                    (r.get("resource_request") or {}).get("memory") for r in rows
                ],
            }
        )
        df_list.append(df)

    if not df_list:
        print("⚠️ instance_events rows empty after load, skipping.")
        return None

    df = pl.concat(df_list, how="diagonal").with_columns(
        (pl.col("time") / 1e6).alias("time_s")
    ).with_columns(
        (pl.col("time_s") // 300 * 300).alias("bucket_s")
    )

    print(f"✅ instance_events loaded: {df.height} rows")
    return df


def load_machine_events(
    folder: str,
    max_files: int | None = None,
    max_lines_per_file: int | None = None,
) -> pl.DataFrame | None:
    """
    Load machine_events shards and return a DataFrame with:

    time (microseconds)
    type (event type)
    machine_id
    cpu_capacity, memory_capacity
    platform_id, clock_rate (if available in schema)

    We'll collapse this into a per-machine capacity summary.
    """
    if not os.path.isdir(folder):
        print(f"⚠️ machine_events folder {folder} not found, skipping.")
        return None

    print(f"📥 Loading machine_events from {folder}")
    files = sorted(glob(os.path.join(folder, "*.json.gz")))
    if not files:
        print("⚠️ No machine_events files found, skipping.")
        return None
    if max_files is not None:
        files = files[:max_files]

    df_list: list[pl.DataFrame] = []
    for path in files:
        print(f"  → {os.path.basename(path)}")
        rows = load_json_gz(path, max_lines_per_file)
        if not rows:
            continue

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
        df_list.append(df)

    if not df_list:
        print("⚠️ machine_events rows empty after load, skipping.")
        return None

    df = pl.concat(df_list, how="diagonal").with_columns(
        (pl.col("time") / 1e6).alias("time_s")
    )

    print(f"✅ machine_events loaded: {df.height} rows")
    return df


# -------------------------------------------------------------------
# Aggregations
# -------------------------------------------------------------------

def summarize_machine_capacity(machine_events: pl.DataFrame | None) -> pl.DataFrame:
    """
    Build a static per-machine capacity summary:

    machine_id, cpu_capacity, memory_capacity, platform_id, clock_rate

    Uses MAX over the event history (ignores REMOVE semantics for simplicity).
    """
    if machine_events is None:
        raise RuntimeError("machine_events is required for machine-level stats")

    print("🧮 Summarizing machine capacities (per machine_id)")

    caps = (
        machine_events
        .group_by("machine_id")
        .agg(
            [
                pl.max("cpu_capacity").alias("cpu_capacity"),
                pl.max("memory_capacity").alias("memory_capacity"),
                pl.max("platform_id").alias("platform_id"),
                pl.max("clock_rate").alias("clock_rate"),
            ]
        )
    )

    print(f"✅ machine capacity summary: {caps.height} machines")
    return caps


def build_machine_level(
    instance_usage: pl.DataFrame,
    machine_caps: pl.DataFrame,
    instance_events: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Build machine-level 5-minute buckets:

    bucket_s, machine_id,
    cpu_used, mem_used,
    num_instances, unique_jobs, unique_instances,
    cpu_capacity, mem_capacity, platform_id, clock_rate,
    utilization_cpu, utilization_mem,
    new_instances (per machine, if instance_events provided)
    """
    print("🧮 Aggregating instance_usage → machine-level buckets")

    # Aggregate per machine & bucket
    usage_agg = (
        instance_usage
        .group_by(["bucket_s", "machine_id"])
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

    # Join with machine capacities
    machine_level = usage_agg.join(machine_caps, on="machine_id", how="left")

    # Optional: per-machine new_instances from instance_events
    if instance_events is not None:
        print("🧮 Aggregating instance_events → per-machine new_instances")
        ev_agg = (
            instance_events
            .group_by(["bucket_s", "machine_id"])
            .agg(pl.count().alias("new_instances_machine"))
        )
        machine_level = machine_level.join(
            ev_agg, on=["bucket_s", "machine_id"], how="left"
        )
        machine_level = machine_level.with_columns(
            pl.col("new_instances_machine").fill_null(0)
        )
    else:
        machine_level = machine_level.with_columns(
            pl.lit(0).alias("new_instances_machine")
        )

    # Utilization metrics
    machine_level = machine_level.with_columns(
        [
            (pl.col("cpu_used") / pl.col("cpu_capacity")).alias("utilization_cpu"),
            (pl.col("mem_used") / pl.col("memory_capacity")).alias("utilization_mem"),
        ]
    )

    print(f"✅ machine-level buckets: {machine_level.height} rows")
    return machine_level


def build_cluster_level(
    machine_level: pl.DataFrame,
    instance_events: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Aggregate machine-level data into cluster-level buckets:

    bucket_s,
    cpu_demand, mem_demand,
    cpu_capacity, mem_capacity,
    machines,
    avg_utilization_cpu, avg_utilization_mem,
    new_instances_cluster (from instance_events)
    """
    print("🧮 Aggregating machine-level → cluster-level")

    cluster = (
        machine_level
        .group_by("bucket_s")
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

    # Cluster-level new instances from instance_events
    if instance_events is not None:
        print("🧮 Aggregating instance_events → cluster-level new_instances")
        ev_cluster = (
            instance_events
            .group_by("bucket_s")
            .agg(pl.count().alias("new_instances_cluster"))
        )
        cluster = cluster.join(ev_cluster, on="bucket_s", how="left")
        cluster = cluster.with_columns(
            pl.col("new_instances_cluster").fill_null(0)
        )
    else:
        cluster = cluster.with_columns(
            pl.lit(0).alias("new_instances_cluster")
        )

    print(f"✅ cluster-level buckets: {cluster.height} rows")
    return cluster


# -------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build machine-level and cluster-level datasets from Google 2019 cluster trace"
    )
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Base directory containing instance_usage, instance_events, machine_events folders")
    parser.add_argument("--out_dir", type=str, default="data/processed",
                        help="Directory to write Parquet outputs")
    parser.add_argument("--max_files_usage", type=int, default=None,
                        help="Max number of instance_usage shards to read (None = all)")
    parser.add_argument("--max_files_events", type=int, default=None,
                        help="Max number of instance_events shards to read (None = all)")
    parser.add_argument("--max_files_machines", type=int, default=None,
                        help="Max number of machine_events shards to read (None = all)")
    parser.add_argument("--max_lines_per_file", type=int, default=None,
                        help="Max lines per JSON.gz file to read (None = all rows)")

    args = parser.parse_args()

    usage_dir = os.path.join(args.raw_dir, "instance_usage")
    events_dir = os.path.join(args.raw_dir, "instance_events")
    machines_dir = os.path.join(args.raw_dir, "machine_events")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all tables
    instance_usage = load_instance_usage(
        usage_dir,
        max_files=args.max_files_usage,
        max_lines_per_file=args.max_lines_per_file,
    )
    instance_events = load_instance_events(
        events_dir,
        max_files=args.max_files_events,
        max_lines_per_file=args.max_lines_per_file,
    )
    machine_events = load_machine_events(
        machines_dir,
        max_files=args.max_files_machines,
        max_lines_per_file=args.max_lines_per_file,
    )

    # Machine capacity summary
    machine_caps = summarize_machine_capacity(machine_events)

    # Machine-level
    machine_level = build_machine_level(
        instance_usage=instance_usage,
        machine_caps=machine_caps,
        instance_events=instance_events,
    )

    machine_out = os.path.join(args.out_dir, "machine_level.parquet")
    print(f"💾 Writing machine-level dataset → {machine_out}")
    machine_level.write_parquet(machine_out)

    # Cluster-level
    cluster_level = build_cluster_level(
        machine_level=machine_level,
        instance_events=instance_events,
    )

    cluster_out = os.path.join(args.out_dir, "cluster_level.parquet")
    print(f"💾 Writing cluster-level dataset → {cluster_out}")
    cluster_level.write_parquet(cluster_out)

    print("🎉 Done.")


if __name__ == "__main__":
    main()
