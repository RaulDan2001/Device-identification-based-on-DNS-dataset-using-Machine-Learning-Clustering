import pandas as pd
import numpy as np
import matplotlib as sns
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle

"""TODO: Update the script to:
        Use CV on device-level features (32 samples) to:
        - Validate feature engineering quality
        - Test clustering stability
        - Ensure features capture device behavior

        Use Train/Test on query-level (230k samples) to:
        - Train final classification model
        - Evaluate real-world performance
        - Build production-ready system"""

class DNSRandomForestClassifier:
    def __init__(self, clustered_devices_path):
        """Initialize Random Forest classifier with clustered devices data"""
        self.clustered_devices_path = clustered_devices_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.rf_model = None
        self.feature_names = None

    def load_data(self):
        """Load clustered devices with features and cluster labels"""
        print(f"INFO: Loading clustered devices from {self.clustered_devices_path}...")
        self.df = pd.read_csv(self.clustered_devices_path)

        print(f"INFO: Loaded {len(self.df)} devices")
        print(f"DEBUG: Columns: {list(self.df.columns)}")
        print(f"DEBUG: Unique devices (MACs): {self.df['mac'].unique()}")
        print(f"DEBUG: Unique clusters: {self.df['cluster'].nunique()}")

        return self.df
    
    def prepare_dataset(self, test_size=0.3, random_state=42, min_sample_per_device=2):
        """Prepare features (X) and labels (y) for classification"""
        if self.df is None:
            raise ValueError("ERROR: Data not loaded. load_data() was not called.")

        print(f"\nINFO: Preparing dataset for classification...")

        # Fillter devices with enough samples for train/test split
        device_counts = self.df['mac'].value_counts()
        valid_devices = device_counts[device_counts >= min_sample_per_device].index

        print(f"INFO: Filltering devices with at least {min_sample_per_device} samples...")
        print(f"INFO: {len(valid_devices)} out of {len(device_counts)} devices have enough samples")

        # Check if we have only 1 row per device
        if len(self.df) == self.df['mac'].nunique():
            print("\nWARNING: Each device has only 1 aggregated feature row.")
            print("INFO: Using cross-validation approach instead of train/test split")

            # Features: all columns except 'mac' (target)
            self.feature_names = [col for col in self.df.columns if col not in ['mac']]
            self.X = self.df[self.feature_names].values

            # Target: MAC addresses (encode them as integers)
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.df['mac'])