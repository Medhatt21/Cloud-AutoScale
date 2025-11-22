# Cloud AutoScale

A clean, minimal baseline autoscaling simulator for cloud infrastructure - Master's Project.

## 🎯 Project Overview

This project implements a **baseline threshold-based autoscaling simulator** that:

- ✅ Loads synthetic demand patterns OR real GCP 2019 trace data
- ✅ Runs discrete-time simulation with configurable autoscaling policies
- ✅ Generates comprehensive visualizations and metrics
- ✅ Focuses ONLY on what's needed for academic evaluation

## 📦 Features

### Data Loading
- **Synthetic Mode**: Generate demand patterns (periodic, bursty, random walk, spike)
- **GCP 2019 Mode**: Load real Google Cloud Platform 2019 trace data

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
# Run with synthetic data (default)
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml

# Run with different patterns
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml --pattern bursty
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml --pattern spike

# Run with GCP 2019 data (after downloading data)
# First, update the config to use mode: "gcp_2019" and set data.path
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml
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

Edit `cloud_autoscale/config/baseline.yaml`:

```yaml
# Mode: "synthetic" or "gcp_2019"
mode: "synthetic"

data:
  path: "data/raw/google"              # For GCP mode
  synthetic_pattern: "periodic"        # periodic, bursty, random_walk, spike
  duration_minutes: 60

simulation:
  step_minutes: 5                      # Time step size
  min_machines: 1
  max_machines: 20
  machine_capacity: 10                 # Units per machine
  cost_per_machine_per_hour: 0.1

autoscaler:
  upper_threshold: 0.7                 # Scale up threshold
  lower_threshold: 0.3                 # Scale down threshold
  max_scale_per_step: 1
  cooldown_steps: 2                    # Prevent thrashing

output:
  directory: "results"
```

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

### Download Data

Use the external scripts provided to download GCP 2019 trace data:

```bash
# Download using the provided scripts
# (scripts should be in scripts/ directory)
bash scripts/download_google_traces.sh
```

### Configure for GCP Mode

Update `cloud_autoscale/config/baseline.yaml`:

```yaml
mode: "gcp_2019"
data:
  path: "data/raw/google"  # Path to downloaded data
  duration_minutes: 60
```

### Run

```bash
uv run cloud-autoscale run --config cloud_autoscale/config/baseline.yaml
```

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

## 🧹 What Was Removed

This refactored version removed:
- ❌ Unused ML pipelines (Prophet, ARIMA, LSTM)
- ❌ Unused RL environments (Stable Baselines)
- ❌ Azure and Alibaba data loaders
- ❌ Complex SimPy-based event simulation
- ❌ Over-engineered abstractions
- ❌ Placeholder classes and unused features
- ❌ Multiple scheduling algorithms (kept only baseline)
- ❌ Excessive boilerplate and defaults

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
