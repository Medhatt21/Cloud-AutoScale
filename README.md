# Cloud Scheduling and Autoscaling Simulator

A comprehensive cloud scheduling and autoscaling simulator implementing baseline, ML/DL, and reinforcement learning approaches for workload management.

## Project Overview

This project implements a multi-stage cloud scheduling simulator:

1. **Stage 1**: Baseline Implementation (AWS/Azure-style scheduling & autoscaling)
2. **Stage 2**: ML/DL-based Scheduling with predictive models
3. **Stage 3**: Reinforcement Learning Scheduling
4. **Stage 4**: Evaluation & Comparison

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

- Python 3.9+
- uv package manager
- Docker (optional, for reproducible environments)

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev

# Install with Jupyter support
uv sync --extra jupyter

# Install with cloud data access
uv sync --extra cloud
```

### Development Setup

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy cloud_scheduler/
```

## Usage

### Running the Simulator

```bash
# Basic simulation with baseline scheduling
uv run cloud-sim simulate --config configs/baseline.yaml

# ML-based scheduling
uv run cloud-sim simulate --config configs/ml_scheduling.yaml

# RL-based scheduling
uv run cloud-sim simulate --config configs/rl_scheduling.yaml

# Evaluation and comparison
uv run cloud-sim evaluate --methods baseline,ml,rl
```

### Data Preparation

```bash
# Download and preprocess Google Cluster Traces
uv run cloud-sim data prepare --dataset google --output data/processed/

# Preprocess Azure Public Dataset
uv run cloud-sim data prepare --dataset azure --output data/processed/
```

## Project Structure

```
cloud-project/
├── cloud_scheduler/           # Main package
│   ├── core/                 # Core simulation engine
│   ├── scheduling/           # Scheduling algorithms
│   ├── ml/                   # ML/DL models
│   ├── rl/                   # Reinforcement learning
│   ├── data/                 # Data processing
│   ├── evaluation/           # Evaluation framework
│   └── utils/                # Utilities
├── configs/                  # Configuration files
├── data/                     # Data storage
├── experiments/              # Experiment results
├── notebooks/                # Jupyter notebooks
├── docker/                   # Docker configurations
└── tests/                    # Test suite
```

## Datasets

- Google Cluster Trace 2019 (BigQuery)
- Azure Public Dataset V2 (2019)
- Alibaba Cluster Trace 2018 (optional)

## Technologies

- **Simulation**: SimPy
- **ML/DL**: PyTorch, scikit-learn, Prophet, ARIMA
- **RL**: Stable-Baselines3, Gymnasium
- **Data**: Polars, Pandas, PyArrow
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Experiment Tracking**: Weights & Biases, MLflow

## License

MIT License
