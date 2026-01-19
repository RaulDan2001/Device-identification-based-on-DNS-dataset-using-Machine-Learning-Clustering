import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DNSFeatureEngineer:
    def __init__(self, dataset_path):
        """Initialize the feature engineer with dataset path"""
        self.dataset_path = dataset_path
        self.df = None
        self.features_df = None
        
    def load_data(self):
        """Load the DNS dataset"""
        print(f"INFO:Loading dataset from {self.dataset_path}...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"INFO: Loaded {len(self.df)} DNS queries")
        print(f"DEBUG: Columns {list(self.df.columns)}")
        return self.df
    
    def explore_data(self, output_dir=None):
        """Create exploratory data analysis plots"""
        if self.df is None:
            raise ValueError("ERROR: Data not loaded. load_data() was not called.")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, "outputs", "exploration")
        
        os.makedirs(output_dir, exist_ok=True)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Distribution of queries per device
        plt.figure(figsize=(12, 6))
        queries_per_device = self.df.groupby('mac').size()
        plt.subplot(1, 2, 1)
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
        domains_per_device = self.df.groupby('mac')['domain'].nunique()
        plt.subplot(1, 2, 1)
        domains_per_device.hist(bins=50, edgecolor='black')
        plt.xlabel('Unique Domains')
        plt.ylabel('Number of MAC addresses')
        plt.title('Distribution of Unique Domains per MAC addreses')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        domains_per_device.plot(kind='box')
        plt.ylabel('Unique Domains')
        plt.title('Unique Domains per Device (Box Plot)')
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
        plt.title('Query Distribution by Day of Week')
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
        """Extract features for each MAC address"""
        if self.df is None:
            raise ValueError("ERROR: Data not loaded. load_data() was not called.")
        
        print("\nEngineering features...")
        
        # Convert timestamp to datetime if not already done
        if 'datetime' not in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
            self.df['hour'] = self.df['datetime'].dt.hour
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        
        features = []
        
        for mac, group in self.df.groupby('mac'):
            device_features = {'mac': mac}
            
            # 1. Basic query statistics
            device_features['total_queries'] = len(group)
            device_features['unique_domains'] = group['domain'].nunique()
            device_features['domain_diversity_ratio'] = device_features['unique_domains'] / device_features['total_queries']
            
            # 2. Query type distribution
            qtype_counts = group['qtype'].value_counts()
            device_features['qtype_1_count'] = qtype_counts.get(1, 0)  # A records
            device_features['qtype_28_count'] = qtype_counts.get(28, 0)  # AAAA records
            device_features['qtype_other_count'] = device_features['total_queries'] - device_features['qtype_1_count'] - device_features['qtype_28_count']
            device_features['qtype_1_ratio'] = device_features['qtype_1_count'] / device_features['total_queries']
            device_features['qtype_28_ratio'] = device_features['qtype_28_count'] / device_features['total_queries']
            
            # 3. Temporal patterns
            device_features['active_hours'] = group['hour'].nunique()
            device_features['active_days'] = group['day_of_week'].nunique()
            device_features['time_span_seconds'] = (group['timestamp'].max() - group['timestamp'].min())
            device_features['time_span_hours'] = device_features['time_span_seconds'] / 3600
            
            # 4. Query frequency
            if device_features['time_span_seconds'] > 0:
                device_features['queries_per_hour'] = device_features['total_queries'] / (device_features['time_span_seconds'] / 3600)
            else:
                device_features['queries_per_hour'] = device_features['total_queries']
            
            # 5. Query interval statistics
            sorted_timestamps = group['timestamp'].sort_values()
            intervals = sorted_timestamps.diff().dropna()
            if len(intervals) > 0:
                device_features['avg_query_interval'] = intervals.mean()
                device_features['std_query_interval'] = intervals.std()
                device_features['min_query_interval'] = intervals.min()
                device_features['max_query_interval'] = intervals.max()
                device_features['median_query_interval'] = intervals.median()
            else:
                device_features['avg_query_interval'] = 0
                device_features['std_query_interval'] = 0
                device_features['min_query_interval'] = 0
                device_features['max_query_interval'] = 0
                device_features['median_query_interval'] = 0
            
            # 6. Domain popularity (how many times top domain is queried)
            domain_counts = group['domain'].value_counts()
            device_features['top_domain_frequency'] = domain_counts.iloc[0] if len(domain_counts) > 0 else 0
            device_features['top_domain_ratio'] = device_features['top_domain_frequency'] / device_features['total_queries']
            
            # 7. Activity concentration (hour of day patterns)
            hour_counts = group['hour'].value_counts()
            device_features['peak_hour_queries'] = hour_counts.max() if len(hour_counts) > 0 else 0
            device_features['peak_hour_ratio'] = device_features['peak_hour_queries'] / device_features['total_queries']
            
            # 8. Entropy of domain distribution (diversity measure)
            domain_probs = domain_counts / domain_counts.sum()
            device_features['domain_entropy'] = -np.sum(domain_probs * np.log2(domain_probs + 1e-10))
            
            # 9. Burstiness (standard deviation of intervals / mean)
            if device_features['avg_query_interval'] > 0:
                device_features['query_burstiness'] = device_features['std_query_interval'] / device_features['avg_query_interval']
            else:
                device_features['query_burstiness'] = 0
            
            # 10. Weekend vs weekday activity
            weekend_queries = len(group[group['day_of_week'].isin([5, 6])])
            device_features['weekend_query_ratio'] = weekend_queries / device_features['total_queries']
            
            features.append(device_features)
        
        self.features_df = pd.DataFrame(features)

        nan_counts = self.features_df.isna().sum()
        if nan_counts.any():
            print("\nWARNING: Found NaN values in features:")
            print(nan_counts[nan_counts > 0])
            print("Filling NaN values with 0...")
            self.features_df = self.features_df.fillna(0.0)

        print(f"INFO: Engineered {len(self.features_df.columns)-1} features for {len(self.features_df)} devices")

        return self.features_df

    def save_features(self, output_path=None):
        """Save engineered features to CSV"""
        if self.features_df is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_path = os.path.join(project_dir, "datasets", "device_features.csv")

        self.features_df.to_csv(output_path, index=False)
        print(f"INFO: Features saved to {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_dir, "datasets", "dns_dataset.csv")
    features_output_path = os.path.join(project_dir, "datasets", "device_features.csv")
    exploration_output_dir = os.path.join(project_dir, "outputs", "exploration")

    os.makedirs(exploration_output_dir, exist_ok=True)
    
    # Initialize feature engineer
    fe = DNSFeatureEngineer(dataset_path)
    
    # Load and explore data
    fe.load_data()
    fe.explore_data(exploration_output_dir)
    
    # Engineer features
    features = fe.engineer_features()
    
    # Display sample features
    print("\nINFO: Sample of engineered features")
    print(features.head())
    print(f"\nINFO: Feature columns: {list(features.columns)}")

    # Save features
    fe.save_features(features_output_path)