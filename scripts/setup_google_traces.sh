#!/bin/bash
# Setup script for downloading Google Cluster Trace data

set -e  # Exit on any error

echo "🌥️  Google Cluster Trace Data Setup"
echo "=================================="

# Create directories
echo "📁 Creating directories..."
mkdir -p data/raw/google
mkdir -p scripts

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK not found!"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Quick install (macOS/Linux):"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo "  exec -l \$SHELL"
    echo ""
    echo "After installation, run this script again."
    echo ""
    echo "Alternative: Install via Homebrew (macOS):"
    echo "  brew install google-cloud-sdk"
    echo ""
    exit 1
fi

echo "✅ Google Cloud SDK found"

# Check authentication
echo "🔐 Checking authentication..."
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -n 1)

if [ -z "$ACTIVE_ACCOUNT" ]; then
    echo "❌ Not authenticated with Google Cloud"
    echo ""
    echo "Please authenticate by running:"
    echo "  gcloud auth login"
    echo "  gcloud auth application-default login"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Authenticated as: $ACTIVE_ACCOUNT"

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
if [ -z "$PROJECT_ID" ]; then
    echo "⚠️  No default project set"
    echo ""
    echo "Please set up a project:"
    echo "1. Create a new project: gcloud projects create YOUR_PROJECT_ID"
    echo "2. Set as default: gcloud config set project YOUR_PROJECT_ID"
    echo ""
    echo "Or use an existing project:"
    echo "  gcloud config set project YOUR_EXISTING_PROJECT_ID"
    echo ""
    echo "Then run this script again."
    exit 1
else
    echo "✅ Current project: $PROJECT_ID"
fi

# Check BigQuery API
echo "🔍 Checking BigQuery API..."
if gcloud services list --enabled --filter="name:bigquery.googleapis.com" --format="value(name)" 2>/dev/null | grep -q bigquery; then
    echo "✅ BigQuery API is enabled"
else
    echo "❌ BigQuery API not enabled"
    echo ""
    echo "Enable it with: gcloud services enable bigquery.googleapis.com"
    read -p "Enable BigQuery API now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Enabling BigQuery API..."
        if gcloud services enable bigquery.googleapis.com; then
            echo "✅ BigQuery API enabled"
        else
            echo "❌ Failed to enable BigQuery API"
            exit 1
        fi
    else
        echo "❌ BigQuery API is required for downloading Google traces"
        echo "Please enable it manually and run this script again."
        exit 1
    fi
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv..."
    echo "Installing cloud dependencies..."
    uv sync --extra cloud
else
    echo "❌ uv package manager not found!"
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or install manually: https://docs.astral.sh/uv/getting-started/installation/"
    echo ""
    echo "After installing uv, restart your shell and run this script again."
    exit 1
fi

echo "✅ Dependencies installed"

# Test BigQuery access
echo "🧪 Testing BigQuery access..."
uv run python -c "
from google.cloud import bigquery
import sys

try:
    client = bigquery.Client()
    # Try to list datasets in the Google cluster data project
    datasets = list(client.list_datasets('google.com:google-cluster-data'))
    print(f'✅ BigQuery access working! Found {len(datasets)} datasets')
    for dataset in datasets[:3]:  # Show first 3
        print(f'  - {dataset.dataset_id}')
    if len(datasets) > 3:
        print(f'  ... and {len(datasets) - 3} more')
except Exception as e:
    print(f'❌ BigQuery access failed: {e}')
    print('Make sure you have:')
    print('1. Authenticated: gcloud auth application-default login')
    print('2. Set a project: gcloud config set project YOUR_PROJECT_ID')
    print('3. Enabled BigQuery API: gcloud services enable bigquery.googleapis.com')
    sys.exit(1)
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download a sample trace:"
echo "   uv run python scripts/download_google_traces.py --trace 2019_05_a --sample-size 100000"
echo ""
echo "2. Available traces:"
echo "   - 2019_05_a (America/New_York)"
echo "   - 2019_05_b (America/Chicago)"  
echo "   - 2019_05_c (America/New_York)"
echo "   - 2019_05_d (America/New_York)"
echo "   - 2019_05_e (Europe/Helsinki)"
echo "   - 2019_05_f (America/Chicago)"
echo "   - 2019_05_g (Asia/Singapore)"
echo "   - 2019_05_h (Europe/Brussels)"
echo ""
echo "3. Monitor your BigQuery usage at:"
echo "   https://console.cloud.google.com/bigquery"
echo ""
echo "💡 Start with a small sample (--sample-size 10000) to test!"
