import pandas as pd
import numpy as np
import os

def enrich_dns_queries():
    """Merge device features and cluster labels back to original DNS queries"""
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    dns_path = os.path.join(project_dir, "datasets", "dns_dataset.csv")
    clustered_path = os.path.join(project_dir, "outputs", "clustering", "clustered_devices.csv")
    output_path = os.path.join(project_dir, "datasets", "enriched_dns_queries.csv")
    
    print("INFO: Loading datasets...")
    
    # Load original DNS queries
    dns_df = pd.read_csv(dns_path)
    print(f"  Loaded {len(dns_df):,} DNS queries")
    
    # Load clustered devices with features
    clustered_df = pd.read_csv(clustered_path)
    print(f"  Loaded {len(clustered_df)} devices with features + clusters")
    
    # Merge: Add device features to each query
    print("\nINFO: Merging device features to queries...")
    enriched_df = dns_df.merge(clustered_df, on='mac', how='left')
    
    # Add query-level temporal features
    print("INFO: Adding query-level temporal features...")
    enriched_df['datetime'] = pd.to_datetime(enriched_df['timestamp'], unit='s')
    enriched_df['hour'] = enriched_df['datetime'].dt.hour
    enriched_df['day_of_week'] = enriched_df['datetime'].dt.dayofweek
    enriched_df['is_weekend'] = enriched_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Drop datetime (keep timestamp)
    enriched_df = enriched_df.drop('datetime', axis=1)
    
    print(f"\nINFO: Enriched dataset created:")
    print(f"  Total queries: {len(enriched_df):,}")
    print(f"  Total columns: {len(enriched_df.columns)}")
    print(f"  Columns: {list(enriched_df.columns)}")
    
    # Check for missing values
    missing = enriched_df.isnull().sum()
    if missing.any():
        print(f"\nWARNING: Missing values found:")
        print(missing[missing > 0])
    
    # Save enriched dataset
    enriched_df.to_csv(output_path, index=False)
    print(f"\nINFO: Saved enriched queries to {output_path}")
    
    return enriched_df

if __name__ == "__main__":
    enriched_df = enrich_dns_queries()