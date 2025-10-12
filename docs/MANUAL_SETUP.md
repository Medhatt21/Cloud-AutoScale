# Manual Setup Guide

If the automated setup script fails, follow these manual steps to set up Google Cloud access for downloading trace data.

## 🔧 Step-by-Step Manual Setup

### 1. Install Google Cloud SDK

#### Option A: Direct Download (Recommended)
```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart your shell
exec -l $SHELL

# Verify installation
gcloud --version
```

#### Option B: Homebrew (macOS)
```bash
brew install google-cloud-sdk
```

#### Option C: Package Manager (Linux)
```bash
# Ubuntu/Debian
sudo apt-get install google-cloud-cli

# CentOS/RHEL
sudo yum install google-cloud-cli
```

### 2. Authenticate with Google Cloud

```bash
# Login to your Google account
gcloud auth login

# Set up application default credentials (required for BigQuery)
gcloud auth application-default login
```

### 3. Set Up a Google Cloud Project

#### Create a New Project
```bash
# Create project (replace with your desired project ID)
gcloud projects create my-cloud-traces-project

# Set as default project
gcloud config set project my-cloud-traces-project
```

#### Or Use Existing Project
```bash
# List your projects
gcloud projects list

# Set existing project as default
gcloud config set project YOUR_EXISTING_PROJECT_ID
```

### 4. Enable BigQuery API

```bash
# Enable the BigQuery API
gcloud services enable bigquery.googleapis.com

# Verify it's enabled
gcloud services list --enabled --filter="name:bigquery.googleapis.com"
```

### 5. Install Python Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install cloud dependencies
uv sync --extra cloud
```

### 6. Test Your Setup

```bash
# Test BigQuery access
uv run python -c "
from google.cloud import bigquery
client = bigquery.Client()
datasets = list(client.list_datasets('google.com:google-cluster-data'))
print(f'✅ Found {len(datasets)} datasets')
"
```

### 7. Download Sample Data

```bash
# Download a small sample (10K records)
uv run python scripts/download_google_traces.py \
  --trace 2019_05_a \
  --sample-size 10000 \
  --output-dir data/raw/google
```

## 🐛 Common Issues and Solutions

### Issue: `gcloud: command not found`

**Solution:**
1. Make sure Google Cloud SDK is installed
2. Restart your terminal/shell
3. Check if gcloud is in your PATH:
   ```bash
   echo $PATH | grep google
   ```
4. If not, add it manually:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export PATH="$HOME/google-cloud-sdk/bin:$PATH"
   source ~/.bashrc  # or ~/.zshrc
   ```

### Issue: Authentication Errors

**Solution:**
```bash
# Clear existing credentials
gcloud auth revoke --all

# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

### Issue: Project Not Set

**Solution:**
```bash
# Check current project
gcloud config get-value project

# Set project
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud config list
```

### Issue: BigQuery API Not Enabled

**Solution:**
```bash
# Enable the API
gcloud services enable bigquery.googleapis.com

# Wait a moment, then verify
gcloud services list --enabled --filter="name:bigquery"
```

### Issue: Permission Denied

**Solution:**
1. Make sure you have the right permissions on the project
2. Check if you're the project owner or have BigQuery permissions
3. Try using a different project where you have admin access

### Issue: Quota Exceeded

**Solution:**
1. Check your BigQuery usage in the [Console](https://console.cloud.google.com/bigquery)
2. Start with smaller sample sizes (--sample-size 1000)
3. BigQuery free tier: 1TB queries/month

## 🔍 Verification Commands

Run these to verify each component:

```bash
# 1. Google Cloud SDK
gcloud --version

# 2. Authentication
gcloud auth list

# 3. Current project
gcloud config get-value project

# 4. BigQuery API
gcloud services list --enabled --filter="name:bigquery"

# 5. Python environment
uv run python -c "import google.cloud.bigquery; print('✅ BigQuery client available')"

# 6. BigQuery access
uv run python -c "
from google.cloud import bigquery
client = bigquery.Client()
print('✅ BigQuery client authenticated')
"
```

## 📞 Getting Help

If you're still having issues:

1. **Check Google Cloud Status**: https://status.cloud.google.com/
2. **Google Cloud Documentation**: https://cloud.google.com/docs
3. **BigQuery Documentation**: https://cloud.google.com/bigquery/docs
4. **Stack Overflow**: Search for "google cloud bigquery authentication"

## 🎯 Quick Test Script

Save this as `test_setup.py` and run it to verify everything works:

```python
#!/usr/bin/env python3
"""Quick test to verify Google Cloud setup."""

def test_setup():
    try:
        from google.cloud import bigquery
        print("✅ BigQuery library imported")
        
        client = bigquery.Client()
        print("✅ BigQuery client created")
        
        # Try to list datasets in the Google cluster data project
        datasets = list(client.list_datasets('google.com:google-cluster-data'))
        print(f"✅ Found {len(datasets)} datasets in Google cluster data project")
        
        if datasets:
            print("Available datasets:")
            for dataset in datasets[:5]:  # Show first 5
                print(f"  - {dataset.dataset_id}")
        
        print("\n🎉 Setup verification successful!")
        print("You can now run the download script.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: uv sync --extra cloud")
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("Check your authentication and project settings.")

if __name__ == "__main__":
    test_setup()
```

Run with:
```bash
uv run python test_setup.py
```
