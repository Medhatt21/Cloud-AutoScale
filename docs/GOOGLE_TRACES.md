# Google Cluster Trace Data Integration

This guide shows you how to download, process, and use Google's cluster trace data for realistic cloud scheduling simulations.

## 🎯 Overview

The Google Cluster Trace v3 dataset contains real workload patterns from Google's production clusters. It includes:

- **Machine configurations** and resource capacities
- **Workload submissions** with resource requirements
- **Resource usage** patterns over time
- **Scheduling events** and state transitions

## 📋 Prerequisites

### 1. Google Cloud Setup

You need a Google Cloud project to access BigQuery:

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth application-default login

# Set your project (or create a new one)
gcloud config set project YOUR_PROJECT_ID

# Enable BigQuery API
gcloud services enable bigquery.googleapis.com
```

### 2. Python Dependencies

The project uses `uv` package manager for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install cloud dependencies (includes BigQuery support)
uv sync --extra cloud

# Or install all optional dependencies
uv sync --all-extras
```

## 🚀 Quick Start

### Step 1: Setup and Authentication

```bash
# Run the setup script
./scripts/setup_google_traces.sh
```

This will:
- ✅ Check Google Cloud SDK installation
- ✅ Verify authentication
- ✅ Enable BigQuery API
- ✅ Install Python dependencies
- ✅ Test BigQuery access

### Step 2: Download Sample Data

```bash
# Download a small sample (10K records per table)
uv run python scripts/download_google_traces.py \
  --trace 2019_05_a \
  --sample-size 10000 \
  --output-dir data/raw/google

# For larger samples (1M records per table)
uv run python scripts/download_google_traces.py \
  --trace 2019_05_a \
  --sample-size 1000000 \
  --output-dir data/raw/google
```

**Available traces:**
- `2019_05_a` - America/New_York
- `2019_05_b` - America/Chicago  
- `2019_05_c` - America/New_York
- `2019_05_d` - America/New_York
- `2019_05_e` - Europe/Helsinki
- `2019_05_f` - America/Chicago
- `2019_05_g` - Asia/Singapore
- `2019_05_h` - Europe/Brussels

### Step 3: Process and Integrate Data

```bash
# Convert raw trace data to simulator format
uv run python scripts/integrate_google_traces.py \
  data/raw/google/2019_05_a \
  --output-dir data/processed/google \
  --duration-hours 1
```

This creates:
- `machines.json` - Machine configurations
- `workloads.json` - Workload patterns  
- `statistics.json` - Summary statistics

### Step 4: Run Simulation

```bash
# Run the complete workflow example
uv run python examples/google_trace_workflow.py

# Or use with the CLI
uv run cloud-sim simulate --config configs/baseline.yaml --data-source google
```

## 📊 Data Structure

### Raw BigQuery Tables

The downloader fetches these tables from BigQuery:

| Table | Description | Key Fields |
|-------|-------------|------------|
| `machine_events` | Machine lifecycle events | `machine_id`, `capacity`, `platform_id` |
| `machine_attributes` | Machine properties | `machine_id`, `name`, `value` |
| `collection_events` | Job/collection lifecycle | `collection_id`, `priority`, `user` |
| `instance_events` | Task/instance events | `collection_id`, `instance_index`, `resource_request` |
| `instance_usage` | Resource usage measurements | `collection_id`, `average_usage`, `maximum_usage` |

### Processed Format

After integration, data is converted to:

#### `machines.json`
```json
[
  {
    "id": "host_12345",
    "cpu_cores": 8,
    "memory_gb": 32.0,
    "zone": "zone_1",
    "instance_type": "t3.xlarge",
    "platform_id": "platform_abc",
    "switch_id": "switch_xyz"
  }
]
```

#### `workloads.json`
```json
[
  {
    "id": "workload_67890_0",
    "collection_id": "67890",
    "arrival_time": 123.45,
    "cpu_request": 2.0,
    "memory_request": 4.0,
    "priority": 120,
    "workload_type": "production",
    "estimated_duration": 600.0,
    "user": "user_hash_xyz"
  }
]
```

## 💰 Cost Management

### BigQuery Free Tier
- **1 TB per month** of query processing
- First 10 GB per month are free
- After that: $5 per TB

### Cost Optimization Tips

1. **Start Small**: Use `--sample-size 10000` for testing
2. **Monitor Usage**: Check your [BigQuery console](https://console.cloud.google.com/bigquery)
3. **Use Processed Data**: Once downloaded, you don't need to re-query
4. **Sample Strategically**: The scripts maintain relationships between tables

### Estimated Costs

| Sample Size | Estimated Query Cost | Monthly Free Tier Usage |
|-------------|---------------------|------------------------|
| 10K records | ~0.1 GB | <1% |
| 100K records | ~1 GB | ~10% |
| 1M records | ~10 GB | ~100% |

## 🔧 Advanced Usage

### Custom Sampling

```python
from scripts.download_google_traces import GoogleTraceDownloader

downloader = GoogleTraceDownloader(project_id="your-project")

# Download specific trace with custom sampling
downloader.download_trace("2019_05_a", "data/raw/google", sample_size=500000)
```

### Data Analysis

```python
from cloud_scheduler.data.loaders import GoogleTraceLoader

loader = GoogleTraceLoader("data/processed/google")
workloads = loader.load_workloads()
machines = loader.load_machines()

# Analyze workload patterns
import pandas as pd
df = pd.DataFrame([{
    'arrival_time': w.arrival_time,
    'cpu_cores': w.resource_specs.cpu_cores,
    'memory_gb': w.resource_specs.memory_gb,
    'duration': w.duration,
    'workload_type': w.workload_type.value
} for w in workloads])

print(df.describe())
```

### Multiple Traces

```bash
# Download multiple geographical traces
for trace in 2019_05_a 2019_05_e 2019_05_g; do
  uv run python scripts/download_google_traces.py \
    --trace $trace \
    --sample-size 100000 \
    --output-dir data/raw/google
done
```

## 🐛 Troubleshooting

### Authentication Issues

```bash
# Re-authenticate
gcloud auth application-default login

# Check active account
gcloud auth list

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### BigQuery Access Errors

```bash
# Enable BigQuery API
gcloud services enable bigquery.googleapis.com

# Check permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### Large Query Costs

If you see high estimated costs:

1. **Reduce sample size**: Use `--sample-size 10000`
2. **Check free tier usage**: Visit [BigQuery console](https://console.cloud.google.com/bigquery)
3. **Use cached results**: Re-running queries uses cache (free)

### Missing Data

If tables are empty:
- Check if the trace exists: `bq show google.com:google-cluster-data:clusterdata_2019_05_a.machine_events`
- Try a different trace (some may have more/less data)
- Verify your sampling parameters

## 📚 References

- [Google Cluster Trace v3 Documentation](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)
- [BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data)
- [BigQuery Pricing](https://cloud.google.com/bigquery/pricing)
- [Google Cloud Free Tier](https://cloud.google.com/free)

## 🤝 Contributing

To add support for other trace formats:

1. Create a new loader class inheriting from `CloudTraceLoader`
2. Implement `load_workloads()` and `load_resource_usage()` methods
3. Add integration scripts following the same pattern
4. Update documentation

Example:
```python
class AzureTraceLoader(CloudTraceLoader):
    def load_workloads(self, limit=None):
        # Implementation for Azure traces
        pass
```
