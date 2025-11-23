# Cloud AutoScale

A production-grade autoscaling simulator for cloud infrastructure - Master's Project.

## 🎯 Project Overview

This project implements a **production-ready autoscaling simulator** that:

- ✅ Loads synthetic demand patterns OR real GCP 2019 trace data (pre-processed)
- ✅ Runs discrete-time simulation with configurable autoscaling policies
- ✅ Generates comprehensive visualizations and metrics
- ✅ **Production-grade**: No defaults, fail-fast validation, explicit configuration
- ✅ Ready for ML and RL extensions (EDA notebook included)

## 📦 Features

### Data Loading
- **Synthetic Mode**: Generate demand patterns (periodic, bursty, random walk, spike)
- **GCP 2019 Mode**: Load pre-processed Google Cloud Platform 2019 trace data from Parquet files
  - Uses `data/processed/cluster_level.parquet` (no raw JSONL processing)
  - Efficient loading with Polars or Pandas
  - No synthetic fallbacks - fails fast if data missing

### Simulation
- Simple discrete-time loop over demand timeline
- Threshold-based autoscaling (scale up/down based on utilization)
- Configurable thresholds, cooldown periods, and capacity limits

### Metrics
- **SLA Violations**: Track when demand exceeds capacity
- **Utilization**: Average, percentiles (P50, P95, P99)
- **Scaling Events**: Count and visualize scale up/down actions
- **Cost**: Simple cost model based on machine hours
- **Stability**: Number of scaling events (lower is better)

### Visualizations
- Demand vs Capacity over time
- Utilization with threshold lines
- Machine count with scaling events
- SLA violations timeline
- Metrics summary dashboard

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** is required
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installation

```bash
# With uv (recommended) - will automatically use Python 3.13
uv sync

# Or manually create venv with Python 3.13 and use pip
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Run Simulation

```bash
# Run with synthetic data
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml

# Run with different patterns
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml --pattern bursty
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml --pattern spike

# Run with GCP 2019 data (after processing data)
# First, ensure processed data exists:
uv run python data/retrievers/process_gcp_data.py

# Then run with GCP config:
uv run cloud-autoscale run --config cloud_autoscale/config/gcp2019.yaml
```

## 📁 Project Structure

```
cloud_autoscale/
├── data/
│   └── loaders/
│       ├── synthetic_loader.py    # Generate synthetic demand patterns
│       └── gcp2019_loader.py      # Load GCP 2019 trace data
├── simulation/
│   ├── simulator.py               # Main discrete-time simulator
│   ├── autoscaler_baseline.py    # Threshold-based autoscaler
│   └── metrics.py                 # Metrics calculation
├── visualization/
│   └── plots.py                   # All visualization functions
├── config/
│   ├── config.py                  # Configuration management
│   └── baseline.yaml              # Default configuration
└── cli/
    └── main.py                    # Command-line interface
```

## ⚙️ Configuration

**⚠️ PRODUCTION MODE: All configuration fields are required. No defaults provided.**

### Synthetic Mode (`cloud_autoscale/config/baseline.yaml`):

```yaml
mode: "synthetic"

data:
  synthetic_pattern: "periodic"        # periodic, bursty, random_walk, spike
  duration_minutes: 60

simulation:
  step_minutes: 5                      # Time step size
  min_machines: 1                      # REQUIRED
  max_machines: 20                     # REQUIRED
  machine_capacity: 10                 # REQUIRED - units per machine
  cost_per_machine_per_hour: 0.1      # REQUIRED

autoscaler:
  upper_threshold: 0.7                 # REQUIRED - scale up threshold
  lower_threshold: 0.3                 # REQUIRED - scale down threshold
  max_scale_per_step: 1               # REQUIRED
  cooldown_steps: 2                    # REQUIRED - prevent thrashing

output:
  directory: "results"                 # REQUIRED
```

### GCP Mode (`cloud_autoscale/config/gcp2019.yaml`):

```yaml
mode: "gcp_2019"

data:
  processed_dir: "data/processed"      # REQUIRED - path to processed Parquet files
  # duration_minutes: 1440             # OPTIONAL - limit simulation duration

simulation:
  step_minutes: 5
  min_machines: 100                    # Larger scale for GCP data
  max_machines: 5000
  machine_capacity: 1
  cost_per_machine_per_hour: 0.05

autoscaler:
  upper_threshold: 0.8
  lower_threshold: 0.4
  max_scale_per_step: 10
  cooldown_steps: 3

output:
  directory: "results"
```

**Missing any required field will cause immediate failure with a clear error message.**

## 📊 Output

Each simulation run creates a timestamped directory in `results/` containing:

- `timeline.csv` - Full simulation timeline data
- `metrics.json` - Calculated metrics
- `config.yaml` - Configuration used for the run
- `plots/` - Directory with all visualizations:
  - `demand_vs_capacity.png`
  - `utilization.png`
  - `machines.png`
  - `violations.png`
  - `metrics_summary.png`

## 🔬 Using GCP 2019 Data

### Step 1: Download Raw Data

Use the provided retrieval script:

```bash
uv run python data/retrievers/get_gcp_data.py
```

This downloads raw JSONL.gz files to `data/raw/`.

### Step 2: Process Data

Convert raw data to processed Parquet files:

```bash
uv run python data/retrievers/process_gcp_data.py
```

This creates:
- `data/processed/cluster_level.parquet` (required for simulation)
- `data/processed/machine_level.parquet` (for EDA)

### Step 3: Run Simulation

```bash
uv run cloud-autoscale run --config cloud_autoscale/config/gcp2019.yaml
```

**Note:** The GCP loader now uses pre-processed Parquet files only. No raw JSONL processing or synthetic fallbacks.

## 📈 Metrics Explained

### SLA Metrics
- **Violation Rate**: Percentage of time steps where demand exceeded capacity
- **Total Violations**: Count of violation events

### Utilization Metrics
- **Average**: Mean utilization across all time steps
- **Percentiles**: P50, P95, P99 utilization values

### Scaling Metrics
- **Scale Up/Down Events**: Count of each type of scaling action
- **Stability Score**: Total scaling events (lower = more stable)

### Cost Metrics
- **Total Cost**: Cumulative cost based on machine hours
- **Cost per Step**: Average cost per time step

### Overall Performance
- **Efficiency Score**: Composite score (0-100) balancing utilization, violations, and stability

## 🔧 Production Refactoring (Latest)

**See `REFACTORING.md` for complete details.**

### What Changed:
- ✅ **No configuration defaults** - all fields must be explicit
- ✅ **Strict validation** - fails fast with clear error messages
- ✅ **GCP loader rewritten** - uses processed Parquet files only
- ✅ **No synthetic fallbacks** - real data or explicit error
- ✅ **Type-safe access** - `config[key]` instead of `config.get(key, default)`
- ✅ **Production-ready** - suitable for deployment and research

### Migration:
- Update config files to include ALL required fields
- For GCP mode: use `processed_dir` instead of `path`
- Run data processing pipeline before using GCP mode
- See example configs: `config/baseline.yaml` and `config/gcp2019.yaml`

## 🧹 What Was Removed (Previous Refactoring)

Earlier cleanup removed:
- ❌ Unused ML pipelines (Prophet, ARIMA, LSTM)
- ❌ Unused RL environments (Stable Baselines)
- ❌ Azure and Alibaba data loaders
- ❌ Complex SimPy-based event simulation
- ❌ Over-engineered abstractions
- ❌ Placeholder classes and unused features
- ❌ Multiple scheduling algorithms (kept only baseline)
- ❌ Excessive boilerplate

## 🎓 Academic Use

This simulator is designed for academic evaluation and comparison of autoscaling policies. It provides:

1. **Reproducible results** with configurable random seeds
2. **Clear metrics** for quantitative comparison
3. **Visual outputs** for qualitative analysis
4. **Simple codebase** suitable for academic submission
5. **Extensible design** for adding new policies

## 📝 License

MIT License

## 👤 Author

Medhata Bouzeid - Master's Project
