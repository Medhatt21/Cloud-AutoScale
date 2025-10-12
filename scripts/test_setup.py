#!/usr/bin/env uv run python
"""Quick test to verify Google Cloud setup."""

def test_setup():
    """Test Google Cloud and BigQuery setup."""
    print("🧪 Testing Google Cloud Setup")
    print("=" * 40)
    
    # Test 1: Import BigQuery library
    try:
        from google.cloud import bigquery
        print("✅ BigQuery library imported")
    except ImportError as e:
        print(f"❌ BigQuery library not found: {e}")
        print("Run: uv sync --extra cloud")
        return False
    
    # Test 2: Create BigQuery client
    try:
        client = bigquery.Client()
        print("✅ BigQuery client created")
    except Exception as e:
        print(f"❌ BigQuery client creation failed: {e}")
        print("Check authentication: gcloud auth application-default login")
        return False
    
    # Test 3: Get current project
    try:
        project = client.project
        print(f"✅ Current project: {project}")
    except Exception as e:
        print(f"❌ Could not get project: {e}")
        return False
    
    # Test 4: Test BigQuery access with Google cluster data
    try:
        datasets = list(client.list_datasets('google.com:google-cluster-data'))
        print(f"✅ Found {len(datasets)} datasets in Google cluster data project")
        
        if datasets:
            print("\nAvailable datasets:")
            for dataset in datasets[:5]:  # Show first 5
                print(f"  - {dataset.dataset_id}")
            if len(datasets) > 5:
                print(f"  ... and {len(datasets) - 5} more")
    except Exception as e:
        print(f"❌ Could not access Google cluster data: {e}")
        print("This might be a permissions issue, but basic BigQuery access works.")
        return True  # This is not a critical failure
    
    # Test 5: Test a simple query
    try:
        query = """
        SELECT COUNT(*) as table_count
        FROM `google.com:google-cluster-data.clusterdata_2019_05_a.INFORMATION_SCHEMA.TABLES`
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(query, job_config=job_config)
        
        bytes_processed = query_job.total_bytes_processed
        print(f"✅ Query validation successful ({bytes_processed} bytes would be processed)")
        
    except Exception as e:
        print(f"⚠️  Query test failed: {e}")
        print("Basic access works, but there might be permission issues with specific datasets.")
    
    print("\n🎉 Setup verification completed!")
    print("\nNext steps:")
    print("1. Download sample data:")
    print("   uv run python scripts/download_google_traces.py --trace 2019_05_a --sample-size 10000")
    print("2. Process the data:")
    print("   uv run python scripts/integrate_google_traces.py data/raw/google/2019_05_a")
    print("3. Run the workflow:")
    print("   uv run python examples/google_trace_workflow.py")
    
    return True

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n🔍 Testing Dependencies")
    print("=" * 30)
    
    required_packages = [
        'google.cloud.bigquery',
        'pandas',
        'numpy',
        'json',
        'pathlib'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🌥️  Cloud Scheduler Setup Test")
    print("=" * 50)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ Some dependencies are missing.")
        print("Run: uv sync --extra cloud")
        exit(1)
    
    # Test Google Cloud setup
    setup_ok = test_setup()
    
    if setup_ok:
        print("\n✅ All tests passed! You're ready to download Google traces.")
    else:
        print("\n❌ Setup issues detected. Please check the manual setup guide:")
        print("   docs/MANUAL_SETUP.md")
        exit(1)
