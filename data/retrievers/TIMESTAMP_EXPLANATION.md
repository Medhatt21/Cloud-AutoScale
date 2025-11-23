# Timestamp Handling in Google Cluster Trace 2019

## Overview

The Google Cluster Trace 2019 dataset does **NOT contain real timestamps**. All time values are relative to an arbitrary trace start point.

## Columns in Processed Data

### 1. `bucket_s` (Float64)
- **Definition**: Seconds elapsed since trace start
- **Range**: 300, 600, 900, ... (5-minute intervals)
- **Usage**: Primary temporal reference for all analysis
- **Example**: `bucket_s=300` means 5 minutes into the trace

### 2. `bucket_index` (Int64)
- **Definition**: 0-based sequential bucket index
- **Calculation**: `(bucket_s / 300) - 1`
- **Range**: 0, 1, 2, 3, ...
- **Usage**: Preferred for temporal analysis, lag features, and indexing
- **Example**: `bucket_index=0` is the first bucket (bucket_s=300)

### 3. `bucket_dt` (Datetime) - **SYNTHETIC**
- **Definition**: Synthetic datetime for visualization convenience only
- **Anchor**: May 1, 2019 00:00:00 UTC
- **Calculation**: `datetime(2019-05-01) + bucket_s seconds`
- **Usage**: **VISUALIZATION ONLY** - do NOT use for temporal analysis
- **Example**: `bucket_s=300` → `bucket_dt=2019-05-01 00:05:00`

## ⚠️ Critical Warnings

### DO NOT Use `bucket_dt` for Analysis
```python
# ❌ WRONG - bucket_dt is synthetic
df.with_columns(pl.col('bucket_dt').dt.hour())

# ✅ CORRECT - use bucket_index
df.with_columns((pl.col('bucket_index') % (24*12)) // 12)  # synthetic hour
```

### DO NOT Extract Date Features from `bucket_dt`
```python
# ❌ WRONG - these are not real dates
df.with_columns([
    pl.col('bucket_dt').dt.day_of_week(),
    pl.col('bucket_dt').dt.month()
])

# ✅ CORRECT - use synthetic cyclical features
df.with_columns([
    (2 * np.pi * pl.col('bucket_index') / 288).sin().alias('sin_day'),
    (2 * np.pi * pl.col('bucket_index') / 288).cos().alias('cos_day')
])
```

## Why `bucket_dt` Exists

`bucket_dt` is provided **ONLY** for:
1. **Human-readable plots**: Easier to read "May 15" than "bucket_s=1209600"
2. **Visualization libraries**: Some tools expect datetime axes
3. **Quick inspection**: Easier to spot patterns visually

## Correct Usage Examples

### ✅ Temporal Analysis
```python
# Use bucket_index for all temporal operations
df.with_columns([
    pl.col('cpu_demand').shift(1).alias('cpu_lag1'),
    pl.col('cpu_demand').rolling_mean(12).alias('cpu_ma_1h')
])
```

### ✅ Seasonality Features
```python
# Create synthetic cycles based on bucket_index
buckets_per_hour = 12
buckets_per_day = 288

df.with_columns([
    (pl.col('bucket_index') % buckets_per_hour).alias('synthetic_hour'),
    (pl.col('bucket_index') % buckets_per_day).alias('synthetic_day')
])
```

### ✅ Visualization (Only Case for bucket_dt)
```python
# OK to use bucket_dt for x-axis labels
plt.plot(df['bucket_dt'], df['cpu_demand'])
plt.xlabel('Time (synthetic, anchored to May 2019)')
```

## Data Processing Pipeline

The `process_gcp_data.py` script:

1. **Reads raw data** with `start_time` and `end_time` in microseconds
2. **Computes bucket_s**: `((start_time + end_time) / 2 / 1e6) // 300 * 300`
3. **Computes bucket_index**: `(bucket_s / 300) - 1`
4. **Adds bucket_dt**: `datetime(2019-05-01) + bucket_s` (for visualization only)

## Summary

| Column | Type | Real? | Use For |
|--------|------|-------|---------|
| `bucket_s` | Float64 | ✅ Yes | Time calculations, sorting |
| `bucket_index` | Int64 | ✅ Yes | **Temporal analysis, ML features** |
| `bucket_dt` | Datetime | ❌ Synthetic | **Visualization only** |

**Golden Rule**: Use `bucket_index` for all analysis. Use `bucket_dt` only for plot labels.

---

**Last Updated**: 2025-11-23
