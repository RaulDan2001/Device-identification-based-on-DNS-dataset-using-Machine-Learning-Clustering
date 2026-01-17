import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import pickle

class DNSKMeansClustering:
    def __init__(self, features_path):
        """Initialize K-Means clustering with features dataset"""
        self.features_path = features_path
        self.df = None
        self.features = None
        self.feature_names = None
        self.scaler = None
        self.scaled_features = None
        self.kmeans = None
        self.optimal_k = None
        
    def load_features(self):
        """Load engineered features"""
        print(f"INFO: Loading features from {self.features_path}...")
        self.df = pd.read_csv(self.features_path)
        
        # Separate MAC addresses and features
        self.feature_names = [col for col in self.df.columns if col != 'mac']
        self.features = self.df[self.feature_names].values
        
        print(f"INFO: Loaded {len(self.df)} devices with {len(self.feature_names)} features")
        print(f"DEBUG: Features: {self.feature_names}")
        
        return self.df
    
    def normalize_features(self):
        """Normalize features using StandardScaler"""
        if self.features is None:
            raise ValueError("ERROR: Features not loaded. load_features() was not called")
        
        print("\nINFO: Normalizing features...")
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        print(f"INFO: Features normalized (mean=0, std=1)")
        
        return self.scaled_features
    
    def find_optimal_k(self, k_range=range(2, 31), output_dir=None):
        """Find optimal K using Elbow Method and Silhouette Score"""
        if self.scaled_features is None:
            raise ValueError("ERROR: Features not normalized. normalize_features() was not called.")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, "outputs", "clustering")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nINFO: Testing K values from {min(k_range)} to {max(k_range)}...")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            print(f"  Testing K = {k}...", end=" ")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, labels))
            
            print(f"Silhouette: {silhouette_scores[-1]:.3f}")
        
        # Plot evaluation metrics
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        
        # 1. Elbow Method
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Silhouette Score (higher is better)
        axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score (Higher = Better)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.5, color='r', linestyle='--', label='Good threshold (0.5)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimal_k_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nINFO: Saved optimal K analysis plot")
        
        # Find optimal K based on Silhouette Score
        optimal_silhouette_idx = np.argmax(silhouette_scores)

        optimal_k_silhouette = list(k_range)[optimal_silhouette_idx]


        print(f"\n{'='*60}")
        print(f"OPTIMAL K RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Based on Silhouette Score: K = {optimal_k_silhouette}")
        print(f"Silhouette Score: {silhouette_scores[optimal_silhouette_idx]:.4f}")
        print(f"{'='*60}\n")
        
        # User input with validation
        min_k, max_k = min(k_range), max(k_range)
        
        while True:
            try:
                user_input = input(f"Select optimal K value ({min_k}-{max_k}) [default: {optimal_k_silhouette}]: ").strip()
                
                # Allow empty input to use default
                if not user_input:
                    self.optimal_k = optimal_k_silhouette
                    print(f"INFO: Using default K = {self.optimal_k}")
                    break
                
                # Validate numeric input
                k_value = int(user_input)
                
                if min_k <= k_value <= max_k:
                    self.optimal_k = k_value
                    print(f"INFO: Selected K = {self.optimal_k}")
                    break
                else:
                    print(f"WARNING: Value must be between {min_k} and {max_k}. Please try again.")
                    
            except ValueError:
                print("ERROR: Invalid input. Please enter a valid integer.")
            except KeyboardInterrupt:
                print("\n\nINFO: Using default K = {optimal_k_silhouette}")
                self.optimal_k = optimal_k_silhouette
                break
        
        print(f"\nINFO: OPTIMAL K is set to: {self.optimal_k}")

        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'k': list(k_range),
            'inertia': inertias,
            'silhouette_score': silhouette_scores
        })
        metrics_df.to_csv(os.path.join(output_dir, 'k_metrics.csv'), index=False)
        print(f"INFO: Saved metrics to k_metrics.csv")
        
        return self.optimal_k
    
    def train_kmeans(self, k=None):
        """Train K-Means model with specified K"""
        if self.scaled_features is None:
            raise ValueError("ERROR: Features not normalized. normalize_features() was not called.")
        
        if k is None:
            if self.optimal_k is None:
                raise ValueError("ERROR: K is not specified. Call find_optimal_k() or provide a K value.")
            k = self.optimal_k

        print(f"\nINFO: Training K-Means with K = {k}...")
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(self.scaled_features)
        
        print(f"INFO: K-Means training complete")
        
        # Cluster distribution
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        print(f"\nCluster Distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} devices ({count/len(self.df)*100:.1f}%)")
        
        return self.kmeans
    
    def visualize_clusters(self, output_dir=None):
        """Visualize clusters using PCA"""
        if self.kmeans is None:
            raise ValueError("ERROR: K-Means not trained. train_kmeans() was not called")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, "outputs", "clustering")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nINFO: Creating cluster visualizations...")
        
        # PCA for 2D visualization
        pca_2d = PCA(n_components=2, random_state=42)
        features_2d = pca_2d.fit_transform(self.scaled_features)
        
        # 2D Scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=self.df['cluster'], cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
        plt.title(f'K-Means Clustering (K={self.kmeans.n_clusters}) - 2D PCA Projection', 
                 fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clusters_2d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved 2D cluster visualization")
        
        # 3D Scatter plot
        pca_3d = PCA(n_components=3, random_state=42)
        features_3d = pca_3d.fit_transform(self.scaled_features)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                           c=self.df['cluster'], cmap='viridis', 
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12)
        ax.set_title(f'K-Means Clustering (K={self.kmeans.n_clusters}) - 3D PCA Projection',
                    fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster', pad=0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clusters_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved 3D cluster visualization")
        
        # Feature importance (cluster centers heatmap)
        cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=self.feature_names
        )
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(cluster_centers.T, annot=True, fmt='.2f', cmap='coolwarm', 
                   cbar_kws={'label': 'Feature Value'}, linewidths=0.5)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Cluster Centers Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_centers_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved cluster centers heatmap")
        
        print(f"\nPCA Explained Variance:")
        print(f"  2D: {sum(pca_2d.explained_variance_ratio_)*100:.2f}%")
        print(f"  3D: {sum(pca_3d.explained_variance_ratio_)*100:.2f}%")
    
    def evaluate_clustering(self):
        """Evaluate clustering quality with metrics"""
        if self.kmeans is None:
            raise ValueError("ERROR: K-Means not trained. train_kmeans() was not called")
        
        print("\n" + "="*60)
        print("CLUSTERING EVALUATION METRICS")
        print("="*60)
        
        silhouette = silhouette_score(self.scaled_features, self.df['cluster'])
        
        print(f"Silhouette Score: {silhouette:.4f} (closer to 1 is better)")
        print(f"Inertia: {self.kmeans.inertia_:.2f}")
        print("="*60)
        
        return {
            'silhouette_score': silhouette,
            'inertia': self.kmeans.inertia_
        }
    
    def save_results(self, output_dir=None):
        """Save clustering results and model"""
        if self.kmeans is None:
            raise ValueError("ERROR: K-Means not trained. train_kmeans() was not called.")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, "outputs", "clustering")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save clustered data with MAC and cluster labels
        output_path = os.path.join(output_dir, 'clustered_devices.csv')
        self.df.to_csv(output_path, index=False)
        print(f"\nINFO: Saved clustered devices to {output_path}")
        
        # Save model and scaler
        model_path = os.path.join(output_dir, 'kmeans_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'optimal_k': self.optimal_k
            }, f)
        print(f"INFO: Saved K-Means model to {model_path}")
        
        return output_path


if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    features_path = os.path.join(project_dir, "datasets", "device_features.csv")
    output_dir = os.path.join(project_dir, "outputs", "clustering")
    
    # Initialize clustering
    kmeans_clustering = DNSKMeansClustering(features_path)
    
    # Load features
    kmeans_clustering.load_features()
    
    # Normalize features
    kmeans_clustering.normalize_features()
    
    # Find optimal K
    optimal_k = kmeans_clustering.find_optimal_k(k_range=range(2, 31))
    
    # Train K-Means with optimal K
    kmeans_clustering.train_kmeans(k=optimal_k)
    
    # Visualize clusters
    kmeans_clustering.visualize_clusters()
    
    # Evaluate clustering
    metrics = kmeans_clustering.evaluate_clustering()
    
    # Save results
    kmeans_clustering.save_results()

    print("\nINFO: K-Means clustering complete")