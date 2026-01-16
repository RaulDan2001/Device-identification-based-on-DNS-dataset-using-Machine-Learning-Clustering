import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DNSFeatureEngineer:
    def __init__(self, dataset_path):
        """Initialize the feature extraction with dataset path"""
        self.dataset_path = dataset_path
        self.df = None
        self.feature_df = None

    def load_data(self):
        "Load the DNS dataset"
        print(f"INFO:Loading dataset from {self.dataset_path}...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"INFO: Loaded {len(self.df)} DNS queries")
        print(f"DEBUG: Columns {list(self.df.columns)}")
        return self.df
    
    def explore_data(self, output_dir='../outputs/exploration'):
        "Create exploratory data analysis plots"
        if self.df is None:
            raise ValueError("ERROR: Data not loaded. load_data() was not called.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        # 1. Distribution of queries per device
        plt.figure(figsize=(12, 6))
        queries_per_device = self.df.groupby('mac').size()
        plt.subplot(1, 2, 2)
        queries_per_device.hist(bins=50, edgecolor='black')
        plt.xlabel('Number of Queries')
        plt.ylabel('Number of Devices')
        plt.title('Distribution of Queries per Device')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        queries_per_device.plot(kind='box')
        plt.ylabel('Number of Queries')
        plt.title('Queries per Device (Box Plot)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_queries_per_device.png'), dpi=300)
        plt.close()
        print("INFO: Saved queries per device plot")

        # 2. Query type distribution
        plt.figure(figsize=(10, 6))
        qtype_counts = self.df['qtype'].value_counts()
        plt.subplot(1, 2, 1)
        qtype_counts.plot(kind='bar', edgecolor='black')
        plt.xlabel('Query Type')
        plt.ylabel('Count')
        plt.title('Query Type Distribution')
        plt.xticks(rotation=0)

        plt.subplot(1, 2, 2)
        qtype_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.ylabel('')
        plt.title('Query Type Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_query_type_distribution.png'), dpi=300)
        plt.close()
        print(f"INFO: Saved query type distribution plot")

        # 3. Unique domains per device
        plt.figure(figsize=(12, 6))
        domains_per_device = self.df.groupby('mac')['domain'].unique()
        plt.subplot(1, 2, 1)
        domains_per_device.hist(bins=50, edgecolor='black')
        plt.xlabel('Unique Domains')
        plt.ylabel('Number of MAC addresses')
        plt.title('Distribution of Unique Domains per MAC addreses')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        domains_per_device.plot(kind='box')
        plt.ylabel('Unique Domains')
        plt.title('Unique Domains per MAC addresses (Box Plot)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_unique_domains_per_device.png'), dpi=300)
        plt.close()
        print(f"INFO: Saved unique domains per MAC addresses plot")

        # 4. Temporal patterns - queries over time
        plt.figure(figsize=(14, 6))
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek

        plt.subplot(1, 2, 1)
        self.df['hour'].hist(bins=24, edgecolor='black')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Queries')
        plt.title('Query Distribution by Hour')
        plt.xticks(range(0, 24))

        plt.subplot(1, 2, 2)
        self.df['day_of_week'].hist(bins=7, edgecolor='black')
        plt.xlabel('Day of Week (0=Monday)')
        plt.ylabel('Number of Queries')
        plt.title("Query Distribution by Day of Week")
        plt.xticks(range(0, 7))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_temporal_patterns.png'), dpi=300)
        plt.close()
        print(f"INFO: Saved temporal patterns plot")

        # 5. Top Domains
        plt.figure(figsize=(12, 6))
        top_domains = self.df['domain'].value_counts().head(20)
        top_domains.plot(kind='barh')
        plt.xlabel('Number of Queries')
        plt.ylabel('Domain (Hashed)')
        plt.title('Top 20 Most Queried Domains')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_top_domains.png'), dpi=300)
        plt.close()
        print(f"INFO: Saved top domains plot")

        # Print summary statistics
        print("\n" + "="*60)
        print("DATASET SUMMARY STATISTICS")
        print("="*60)
        print(f"Total queries: {len(self.df):,}")
        print(f"Unique devices (MACs): {self.df['mac'].nunique():,}")
        print(f"Unique domains: {self.df['domain'].nunique():,}")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        print(f"\nQueries per device - Mean: {queries_per_device.mean():.2f}, Median: {queries_per_device.median():.2f}")
        print(f"Domains per device - Mean: {domains_per_device.mean():.2f}, Median: {domains_per_device.median():.2f}")
        print("="*60)

    def engineer_features(self):
        pass #TODO: implement feature engineering

    def save_features(self):
        pass #TODO: implement save features to file

if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_dir, "datasets", "dns_dataset.csv")

    # Initialize features engineer
    fe = DNSFeatureEngineer(dataset_path)

    # Load and explore data
    fe.load_data()
    fe.explore_data()

    # Engineer features
    features = fe.engineer_features()

    # Display sample features
    print("\INFO: Sample of engineered features")
    print(features.head())
    print(f"\INFO: Feature columns: {list(features.columns)}")

    # Save features
    fe.save_features()