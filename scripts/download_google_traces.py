#!/usr/bin/env uv run python
"""
Google Cluster Trace Data Downloader
Downloads sampled Google cluster trace data from BigQuery while maintaining relationships.
Respects BigQuery free tier limitations (1TB per month query limit).
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account
import logging
from typing import Dict, List, Set
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleTraceDownloader:
    def __init__(self, project_id: str = None, credentials_path: str = None):
        """
        Initialize the downloader.
        
        Args:
            project_id: Your Google Cloud project ID (if None, uses default)
            credentials_path: Path to service account JSON (if None, uses default auth)
        """
        self.project_id = project_id
        
        # Initialize BigQuery client
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            # Use default credentials (gcloud auth application-default login)
            self.client = bigquery.Client(project=project_id)
        
        # Available traces
        self.available_traces = [
            '2019_05_a', '2019_05_b', '2019_05_c', '2019_05_d',
            '2019_05_e', '2019_05_f', '2019_05_g', '2019_05_h'
        ]
        
        # Table definitions with their relationships
        self.tables = {
            'machine_events': {
                'primary_key': ['time', 'machine_id'],
                'sample_strategy': 'random',
                'relationships': []
            },
            'machine_attributes': {
                'primary_key': ['time', 'machine_id', 'name'],
                'sample_strategy': 'related',  # Sample based on machines from machine_events
                'relationships': ['machine_events.machine_id']
            },
            'collection_events': {
                'primary_key': ['time', 'collection_id'],
                'sample_strategy': 'random',
                'relationships': []
            },
            'instance_events': {
                'primary_key': ['time', 'collection_id', 'instance_index'],
                'sample_strategy': 'related',  # Sample based on collections
                'relationships': ['collection_events.collection_id']
            },
            'instance_usage': {
                'primary_key': ['start_time', 'collection_id', 'instance_index'],
                'sample_strategy': 'related',  # Sample based on instances
                'relationships': ['instance_events.collection_id', 'instance_events.instance_index']
            }
        }
        
        self.sampled_entities = {}  # Track sampled entities to maintain relationships
    
    def estimate_query_cost(self, query: str) -> float:
        """Estimate query cost in GB processed."""
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = self.client.query(query, job_config=job_config)
        return query_job.total_bytes_processed / (1024**3)  # Convert to GB
    
    def get_sample_query(self, trace: str, table: str, sample_size: int = 1000000) -> str:
        """Generate sampling query based on table type and relationships."""
        dataset = f"google.com:google-cluster-data:clusterdata_{trace}"
        table_info = self.tables[table]
        
        if table_info['sample_strategy'] == 'random':
            # Random sampling for base tables
            if table == 'machine_events':
                # Sample machines first, then get all their events
                query = f"""
                WITH sampled_machines AS (
                  SELECT DISTINCT machine_id
                  FROM `{dataset}.{table}`
                  WHERE RAND() < {sample_size / 10000000}  -- Adjust based on expected table size
                  LIMIT {sample_size // 100}  -- Limit number of unique machines
                )
                SELECT me.*
                FROM `{dataset}.{table}` me
                INNER JOIN sampled_machines sm ON me.machine_id = sm.machine_id
                LIMIT {sample_size}
                """
            elif table == 'collection_events':
                # Sample collections first, then get all their events
                query = f"""
                WITH sampled_collections AS (
                  SELECT DISTINCT collection_id
                  FROM `{dataset}.{table}`
                  WHERE RAND() < {sample_size / 10000000}  -- Adjust based on expected table size
                  LIMIT {sample_size // 50}  -- Limit number of unique collections
                )
                SELECT ce.*
                FROM `{dataset}.{table}` ce
                INNER JOIN sampled_collections sc ON ce.collection_id = sc.collection_id
                LIMIT {sample_size}
                """
            else:
                # Fallback random sampling
                query = f"""
                SELECT *
                FROM `{dataset}.{table}`
                WHERE RAND() < {sample_size / 10000000}
                LIMIT {sample_size}
                """
        
        elif table_info['sample_strategy'] == 'related':
            # Sample based on relationships
            if table == 'machine_attributes':
                # Use sampled machine_ids
                machine_ids = self.sampled_entities.get('machine_ids', [])
                if machine_ids:
                    machine_id_list = "', '".join(str(mid) for mid in machine_ids[:1000])  # Limit for query size
                    query = f"""
                    SELECT *
                    FROM `{dataset}.{table}`
                    WHERE machine_id IN ('{machine_id_list}')
                    LIMIT {sample_size}
                    """
                else:
                    query = f"SELECT * FROM `{dataset}.{table}` WHERE FALSE"  # Empty result
            
            elif table == 'instance_events':
                # Use sampled collection_ids
                collection_ids = self.sampled_entities.get('collection_ids', [])
                if collection_ids:
                    collection_id_list = "', '".join(str(cid) for cid in collection_ids[:1000])
                    query = f"""
                    SELECT *
                    FROM `{dataset}.{table}`
                    WHERE collection_id IN ('{collection_id_list}')
                    LIMIT {sample_size}
                    """
                else:
                    query = f"SELECT * FROM `{dataset}.{table}` WHERE FALSE"
            
            elif table == 'instance_usage':
                # Use sampled collection_ids and instance_indices
                collection_ids = self.sampled_entities.get('collection_ids', [])
                if collection_ids:
                    collection_id_list = "', '".join(str(cid) for cid in collection_ids[:500])  # Smaller limit for usage data
                    query = f"""
                    SELECT *
                    FROM `{dataset}.{table}`
                    WHERE collection_id IN ('{collection_id_list}')
                    LIMIT {sample_size}
                    """
                else:
                    query = f"SELECT * FROM `{dataset}.{table}` WHERE FALSE"
            
            else:
                # Fallback
                query = f"SELECT * FROM `{dataset}.{table}` LIMIT {sample_size}"
        
        return query
    
    def download_table(self, trace: str, table: str, sample_size: int = 1000000) -> pd.DataFrame:
        """Download a table with sampling."""
        logger.info(f"Downloading {table} from trace {trace} (target: {sample_size:,} records)")
        
        query = self.get_sample_query(trace, table, sample_size)
        
        # Estimate cost
        estimated_gb = self.estimate_query_cost(query)
        logger.info(f"Estimated query cost: {estimated_gb:.2f} GB")
        
        if estimated_gb > 50:  # Safety check
            logger.warning(f"Query cost is high ({estimated_gb:.2f} GB). Consider reducing sample size.")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                return pd.DataFrame()
        
        # Execute query
        start_time = time.time()
        df = self.client.query(query).to_dataframe()
        duration = time.time() - start_time
        
        logger.info(f"Downloaded {len(df):,} records from {table} in {duration:.1f}s")
        
        # Store sampled entities for relationship maintenance
        if table == 'machine_events' and 'machine_id' in df.columns:
            self.sampled_entities['machine_ids'] = df['machine_id'].unique().tolist()
            logger.info(f"Stored {len(self.sampled_entities['machine_ids'])} unique machine IDs")
        
        elif table == 'collection_events' and 'collection_id' in df.columns:
            self.sampled_entities['collection_ids'] = df['collection_id'].unique().tolist()
            logger.info(f"Stored {len(self.sampled_entities['collection_ids'])} unique collection IDs")
        
        elif table == 'instance_events':
            if 'collection_id' in df.columns and 'instance_index' in df.columns:
                instances = list(zip(df['collection_id'], df['instance_index']))
                self.sampled_entities['instances'] = instances
                logger.info(f"Stored {len(instances)} instance identifiers")
        
        return df
    
    def download_trace(self, trace: str, output_dir: str, sample_size: int = 1000000):
        """Download a complete trace with all tables."""
        logger.info(f"Starting download of trace {trace}")
        
        if trace not in self.available_traces:
            logger.error(f"Trace {trace} not available. Available: {self.available_traces}")
            return
        
        output_path = Path(output_dir) / trace
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Clear sampled entities for this trace
        self.sampled_entities = {}
        
        # Download tables in dependency order
        download_order = ['machine_events', 'machine_attributes', 'collection_events', 'instance_events', 'instance_usage']
        
        metadata = {
            'trace': trace,
            'sample_size': sample_size,
            'tables': {},
            'relationships_maintained': True
        }
        
        for table in download_order:
            try:
                df = self.download_table(trace, table, sample_size)
                
                if not df.empty:
                    # Save as parquet for efficiency
                    output_file = output_path / f"{table}.parquet"
                    df.to_parquet(output_file, index=False)
                    
                    # Also save as CSV for compatibility
                    csv_file = output_path / f"{table}.csv"
                    df.to_csv(csv_file, index=False)
                    
                    metadata['tables'][table] = {
                        'records': len(df),
                        'columns': list(df.columns),
                        'file_size_mb': output_file.stat().st_size / (1024*1024)
                    }
                    
                    logger.info(f"Saved {table}: {len(df):,} records to {output_file}")
                else:
                    logger.warning(f"No data downloaded for {table}")
                    metadata['tables'][table] = {'records': 0, 'columns': [], 'file_size_mb': 0}
            
            except Exception as e:
                logger.error(f"Failed to download {table}: {e}")
                metadata['tables'][table] = {'error': str(e)}
        
        # Save metadata
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Trace {trace} download completed. Metadata saved to {metadata_file}")
        
        # Print summary
        total_records = sum(t.get('records', 0) for t in metadata['tables'].values())
        total_size = sum(t.get('file_size_mb', 0) for t in metadata['tables'].values())
        logger.info(f"Total: {total_records:,} records, {total_size:.1f} MB")

def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Google Cluster Trace data')
    parser.add_argument('--trace', default='2019_05_a', choices=[
        '2019_05_a', '2019_05_b', '2019_05_c', '2019_05_d',
        '2019_05_e', '2019_05_f', '2019_05_g', '2019_05_h'
    ], help='Trace to download')
    parser.add_argument('--output-dir', default='data/raw/google', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=1000000, help='Records per table')
    parser.add_argument('--project-id', help='Google Cloud project ID')
    parser.add_argument('--credentials', help='Path to service account JSON')
    
    args = parser.parse_args()
    
    try:
        downloader = GoogleTraceDownloader(args.project_id, args.credentials)
        downloader.download_trace(args.trace, args.output_dir, args.sample_size)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Make sure you have:")
        logger.info("1. Google Cloud SDK installed: https://cloud.google.com/sdk/docs/install")
        logger.info("2. Authenticated: gcloud auth application-default login")
        logger.info("3. BigQuery API enabled in your project")
        sys.exit(1)

if __name__ == '__main__':
    main()
