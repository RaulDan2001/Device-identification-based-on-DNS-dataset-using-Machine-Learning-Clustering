import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle

class DNSRandomForestClassifier:
    def __init__(self, enriched_queries_path, clustered_devices_path):
        """Initialize Random Forest classifier with enriched queries and clustered devices"""
        self.enriched_queries_path = enriched_queries_path
        self.clustered_devices_path = clustered_devices_path
        self.queries_df = None
        self.devices_df = None
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
        """Load enriched queries and clustered devices"""
        print(f"INFO: Loading enriched queries from {self.enriched_queries_path}...")
        self.queries_df = pd.read_csv(self.enriched_queries_path)
        
        print(f"INFO: Loading clustered devices from {self.clustered_devices_path}...")
        self.devices_df = pd.read_csv(self.clustered_devices_path)
        
        print(f"\nINFO: Dataset loaded:")
        print(f"  Enriched queries: {len(self.queries_df):,} rows")
        print(f"  Devices: {len(self.devices_df)} devices")
        print(f"  Query columns: {len(self.queries_df.columns)}")
        print(f"  Unique MACs in queries: {self.queries_df['mac'].nunique()}")
        
        return self.queries_df, self.devices_df
    
    def prepare_query_level_dataset(self, test_size=0.3, random_state=42, min_queries_per_device=2):
        """
        Prepare query-level dataset for train/test split
        Uses enriched queries (230k+ samples) for robust evaluation
        """
        if self.queries_df is None:
            raise ValueError("ERROR: Data not loaded. Call load_data() first.")
        
        print(f"\n{'='*70}")
        print("QUERY-LEVEL DATASET PREPARATION")
        print(f"{'='*70}")
        
        # Check device distribution
        device_counts = self.queries_df['mac'].value_counts()
        print(f"\nINFO: Device query distribution:")
        print(f"  Total devices: {len(device_counts)}")
        print(f"  Total queries: {len(self.queries_df):,}")
        print(f"  Min queries per device: {device_counts.min()}")
        print(f"  Max queries per device: {device_counts.max():,}")
        print(f"  Median queries per device: {device_counts.median():.0f}")
        
        # Filter devices with too few queries for stratified split
        devices_too_few = device_counts[device_counts < min_queries_per_device]
        if len(devices_too_few) > 0:
            print(f"\nWARNING: {len(devices_too_few)} devices have < {min_queries_per_device} queries:")
            print(f"  Devices: {list(devices_too_few.index)[:5]}..." if len(devices_too_few) > 5 else f"  Devices: {list(devices_too_few.index)}")
            print(f"  Removing these devices for reliable train/test split...")
            
            # Filter out devices with too few samples
            valid_devices = device_counts[device_counts >= min_queries_per_device].index
            original_len = len(self.queries_df)
            self.queries_df = self.queries_df[self.queries_df['mac'].isin(valid_devices)]
            removed_queries = original_len - len(self.queries_df)
            
            print(f"  Removed {removed_queries:,} queries ({removed_queries/original_len*100:.2f}%)")
            print(f"  Remaining: {len(self.queries_df):,} queries from {len(valid_devices)} devices")
        
        # Define features: exclude identifiers and target
        exclude_cols = ['mac', 'domain', 'timestamp']
        self.feature_names = [col for col in self.queries_df.columns if col not in exclude_cols]
        
        print(f"\nINFO: Preparing query-level features...")
        print(f"  Total features: {len(self.feature_names)}")
        print(f"  Feature categories:")
        print(f"    - Query-level: qtype, hour, day_of_week, is_weekend")
        print(f"    - Device-level: {[f for f in self.feature_names if f not in ['qtype', 'hour', 'day_of_week', 'is_weekend', 'cluster']][:5]}...")
        print(f"    - Cluster label: cluster")
        
        # Features (X)
        self.X = self.queries_df[self.feature_names].values
        
        # Target (y): MAC addresses
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.queries_df['mac'])
        
        # Check if stratified split is possible
        unique, counts = np.unique(self.y, return_counts=True)
        can_stratify = all(counts >= 2)
        
        # Train/Test split
        print(f"\nINFO: Splitting dataset (test_size={test_size})...")
        if can_stratify:
            print(f"  Using stratified split (all devices have >= {min_queries_per_device} queries)")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y,
                test_size=test_size,
                random_state=random_state,
                stratify=self.y
            )
        else:
            print(f"  WARNING: Using non-stratified split (some devices still have few queries)")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y,
                test_size=test_size,
                random_state=random_state,
                stratify=None
            )
        
        print(f"\nINFO: Dataset split complete:")
        print(f"  Training set: {len(self.X_train):,} queries ({len(self.X_train)/len(self.X)*100:.1f}%)")
        print(f"  Test set: {len(self.X_test):,} queries ({len(self.X_test)/len(self.X)*100:.1f}%)")
        print(f"  Number of features: {self.X_train.shape[1]}")
        print(f"  Number of device classes: {len(self.label_encoder.classes_)}")
        print(f"  Queries per device (avg): {len(self.queries_df) / len(self.label_encoder.classes_):.0f}")
        
        # Check class distribution in splits
        train_devices = len(np.unique(self.y_train))
        test_devices = len(np.unique(self.y_test))
        print(f"  Devices in training set: {train_devices}")
        print(f"  Devices in test set: {test_devices}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        """Train Random Forest classifier on query-level data"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("ERROR: Dataset not prepared. Call prepare_query_level_dataset() first.")
        
        print(f"\n{'='*70}")
        print("TRAINING RANDOM FOREST ON QUERY-LEVEL DATA")
        print(f"{'='*70}")
        
        print(f"\nINFO: Training Random Forest Classifier...")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth if max_depth else 'None (unlimited)'}")
        print(f"  min_samples_split: {min_samples_split}")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        
        print(f"\nINFO:Random Forest training complete!")
        
        return self.rf_model
    
    def evaluate_model(self, output_dir=None):
        """Evaluate Random Forest model performance on query-level data"""
        if self.rf_model is None:
            raise ValueError("ERROR: Model not trained. Call train_random_forest() first.")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, "outputs", "random_forest")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("PHASE 4: QUERY-LEVEL MODEL EVALUATION (Train/Test Split)")
        print(f"{'='*70}")
        
        # Predictions
        print("\nINFO: Making predictions...")
        y_train_pred = self.rf_model.predict(self.X_train)
        y_test_pred = self.rf_model.predict(self.X_test)
        
        # Accuracy
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        # F1 Scores
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        print(f"\n{'='*70}")
        print("QUERY-LEVEL TRAIN/TEST RESULTS:")
        print(f"{'='*70}")
        print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Training F1-Score: {train_f1:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        
        # Check for overfitting
        overfit_gap = train_accuracy - test_accuracy
        if overfit_gap > 0.1:
            print(f" WARNING:Overfitting detected (gap: {overfit_gap:.2%})")
        else:
            print(f" SUCCESS:Model generalizes well (gap: {overfit_gap:.2%})")
        
        print(f"{'='*70}")
        
        # Classification Report
        print(f"\nClassification Report (Test Set - Top 10 Devices):")
        class_names = self.label_encoder.classes_
        
        # Get top 10 most common devices in test set
        unique, counts = np.unique(self.y_test, return_counts=True)
        top_10_indices = unique[np.argsort(counts)[-10:]]
        top_10_classes = [class_names[i] for i in top_10_indices]
        
        # Filter predictions for top 10 classes
        mask = np.isin(self.y_test, top_10_indices)
        report = classification_report(
            self.y_test[mask],
            y_test_pred[mask],
            labels=top_10_indices,
            target_names=top_10_classes,
            zero_division=0
        )
        print(report)
        
        # Save full classification report
        full_report = classification_report(
            self.y_test, y_test_pred,
            target_names=[str(c) for c in class_names],
            zero_division=0
        )
        
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write("RANDOM FOREST CLASSIFICATION REPORT (QUERY-LEVEL)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Training Set: {len(self.X_train):,} queries\n")
            f.write(f"Test Set: {len(self.X_test):,} queries\n\n")
            f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Training F1-Score: {train_f1:.4f}\n")
            f.write(f"Test F1-Score: {test_f1:.4f}\n\n")
            f.write(full_report)
        
        print(f"\nINFO: Saved full classification report to {output_dir}")
        
        # Confusion Matrix (for top 10 devices only - 32x32 is too large)
        print(f"\nINFO: Generating confusion matrix for top 10 devices...")
        cm = confusion_matrix(self.y_test[mask], y_test_pred[mask], labels=top_10_indices)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[c[:16] for c in top_10_classes],  # Truncate long names
                   yticklabels=[c[:16] for c in top_10_classes],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Device', fontsize=12)
        plt.ylabel('True Device', fontsize=12)
        plt.title('Confusion Matrix - Top 10 Devices (Query-Level Classification)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_top10.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved confusion matrix")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'overfit_gap': overfit_gap
        }
    
    def analyze_feature_importance(self, output_dir=None, top_n=25):
        """Analyze and visualize feature importance"""
        if self.rf_model is None:
            raise ValueError("ERROR: Model not trained. Call train_random_forest() first.")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, "outputs", "random_forest")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*70}")
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': [self.feature_names[i] for i in indices],
            'importance': importances[indices],
            'importance_pct': importances[indices] / importances.sum() * 100
        })
        
        # Save to CSV
        feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print(f"INFO: Saved feature importance to CSV")
        
        # Print top features
        print(f"\nTop {top_n} Most Important Features:")
        print(f"{'='*70}")
        for idx, row in feature_importance_df.head(top_n).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:30s}: {row['importance']:.4f} ({row['importance_pct']:.2f}%)")
        print(f"{'='*70}")
        
        # Visualize top N features
        plt.figure(figsize=(14, 10))
        top_features = feature_importance_df.head(top_n)
        colors = ['red' if 'cluster' in f else 'steelblue' for f in top_features['feature']]
        
        plt.barh(range(top_n), top_features['importance'].values[::-1], 
                color=colors[::-1], edgecolor='black', linewidth=0.5)
        plt.yticks(range(top_n), top_features['feature'].values[::-1])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features for Device Classification (Query-Level)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', edgecolor='black', label='Original Features'),
            Patch(facecolor='red', edgecolor='black', label='Cluster Feature')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved feature importance plot")
        
        # Analyze cluster feature importance
        cluster_row = feature_importance_df[feature_importance_df['feature'] == 'cluster']
        if not cluster_row.empty:
            cluster_rank = cluster_row.index[0] + 1
            cluster_value = cluster_row['importance'].values[0]
            cluster_pct = cluster_row['importance_pct'].values[0]
            
            print(f"\n{'='*70}")
            print("CLUSTER FEATURE ANALYSIS:")
            print(f"{'='*70}")
            print(f"  Rank: {cluster_rank} out of {len(self.feature_names)}")
            print(f"  Importance: {cluster_value:.4f}")
            print(f"  Percentage: {cluster_pct:.2f}%")
        
        return feature_importance_df
    
    def save_model(self, output_dir=None):
        """Save trained Random Forest model"""
        if self.rf_model is None:
            raise ValueError("ERROR: Model not trained. Call train_random_forest() first.")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, "outputs", "random_forest")
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, 'random_forest_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.rf_model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, f)
        
        print(f"\nINFO:Saved Random Forest model to {model_path}")
        
        return model_path


if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    enriched_queries_path = os.path.join(project_dir, "datasets", "enriched_dns_queries.csv")
    clustered_devices_path = os.path.join(project_dir, "outputs", "clustering", "clustered_devices.csv")
    output_dir = os.path.join(project_dir, "outputs", "random_forest")
    
    # Initialize Random Forest classifier
    print("="*70)
    print("RANDOM FOREST DEVICE FINGERPRINTING PIPELINE")
    print("="*70)
    
    rf_classifier = DNSRandomForestClassifier(enriched_queries_path, clustered_devices_path)
    
    # Load data
    rf_classifier.load_data()
    
    # PHASE 2: Prepare query-level dataset
    rf_classifier.prepare_query_level_dataset(test_size=0.3, random_state=42)
    
    # PHASE 3: Train Random Forest on query-level data
    rf_classifier.train_random_forest(n_estimators=100, max_depth=20, random_state=42)
    
    # PHASE 4: Evaluate model
    metrics = rf_classifier.evaluate_model()
    
    # PHASE 5: Analyze feature importance
    feature_importance = rf_classifier.analyze_feature_importance(top_n=25)
    
    # Save model
    rf_classifier.save_model()
    
    print("\n" + "="*70)
    print("INFO:RANDOM FOREST PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nSUMMARY:")
    #print(f"  Device-level CV Accuracy: {cv_results['mean_accuracy']:.2%} ± {cv_results['std_accuracy']:.2%}")
    print(f"  Query-level Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  Query-level Test F1-Score: {metrics['test_f1']:.4f}")
    print("="*70)