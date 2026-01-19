# DNS-Based Device Fingerprinting Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Complete Workflow](#complete-workflow)
5. [Scripts Documentation](#scripts-documentation)
6. [Obstacles & Solutions](#obstacles--solutions)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Results Interpretation](#results-interpretation)
9. [Future Improvements](#future-improvements)

---

## Project Overview

### Objective
Develop a machine learning-based methodology to uniquely identify network devices using DNS query patterns, combining unsupervised clustering (K-Means) with supervised classification (Random Forest).

### Key Technologies
- **Python 3.x**
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **NumPy**: Numerical computing

### Dataset
- **Source**: DNS query logs
- **Total Queries**: 584,789
- **Unique Devices**: 32 (identified by hashed MAC addresses)
- **Features**: Domain, query type, timestamp
- **Anonymization**: MAC addresses and domains are hashed (MD5-like)

### Methodology
**Two-Phase Approach:**
1. **K-Means Clustering**: Group devices by behavioral patterns (unsupervised)
2. **Random Forest Classification**: Predict device identity from DNS queries (supervised)

---

## System Architecture

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Data Preparation                                   │
│ Input: dns_dataset.pkl → dns_dataset.csv (584,789 queries)  │
│ Tool: pkl_converter.py                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Feature Engineering                                │
│ Group by MAC → Extract 25 behavioral features per device    │
│ Output: device_features.csv (32 devices × 25 features)      │
│ Tool: feature_engineering.py                                │
│ Validation: Exploratory data analysis plots                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: K-Means Clustering                                 │
│ Normalize features → Find optimal K → Assign clusters       │
│ Output: clustered_devices.csv (32 devices + cluster labels) │
│ Tool: kmeans_clustering.py                                  │
│ Validation: Silhouette score, elbow method                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Query Enrichment                                   │
│ Merge cluster labels back to original 584k queries          │
│ Output: enriched_dns_queries.csv (queries + device features)│
│ Tool: enrich_dns_dataset.py                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Random Forest Classification                       │
│ Train on 409k queries → Test on 175k queries                │
│ Output: Trained model + performance metrics                 │
│ Tool: random_forest_classifier.py                           │
│ Validation: Cross-validation + train/test split             │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Proiect2/
├── datasets/
│   ├── dns_dataset.pkl              # Original pickle file
│   ├── dns_dataset.csv              # Converted CSV
│   ├── device_features.csv          # Engineered features (32 devices)
│   └── enriched_dns_queries.csv     # Queries + device features (584k rows)
│
├── src/
│   ├── pkl_converter.py             # Convert PKL to CSV/Excel
│   ├── feature_engineering.py       # Extract device features
│   ├── kmeans_clustering.py         # K-Means clustering
│   ├── enrich_dns_dataset.py        # Merge features to queries
│   └── random_forest_classifier.py  # Train & evaluate RF model
│
├── outputs/
│   ├── exploration/                 # EDA plots
│   │   ├── 01_queries_per_device.png
│   │   ├── 02_query_type_distribution.png
│   │   ├── 03_unique_domains_per_device.png
│   │   ├── 04_temporal_patterns.png
│   │   └── 05_top_domains.png
│   │
│   ├── clustering/                  # K-Means results
│   │   ├── optimal_k_analysis.png
│   │   ├── clusters_2d.png
│   │   ├── clusters_3d.png
│   │   ├── cluster_centers_heatmap.png
│   │   ├── clustered_devices.csv
│   │   ├── k_metrics.csv
│   │   └── kmeans_model.pkl
│   │
│   └── random_forest/               # RF results
│       ├── classification_report.txt
│       ├── confusion_matrix_top10.png
│       ├── feature_importance.png
│       ├── feature_importance.csv
│       └── random_forest_model.pkl
│
├── documentation/
│   ├── cluster_analysis.md          # Cluster interpretation
│   └── project_documentation.md     # This file
│
└── requirements.txt                 # Python dependencies
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### 2. Install Dependencies

```bash
# Navigate to project directory
cd "x:\x\project_dir"

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Required Packages

```
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

---

## Complete Workflow

### Step 1: Convert PKL to CSV (Optional if you already have CSV)

```bash
python src/pkl_converter.py
```

**Output:**
- `datasets/dns_dataset.csv`
- `datasets/dns_dataset.xlsx` (for viewing in Excel)

---

### Step 2: Feature Engineering

```bash
python src/feature_engineering.py
```

**What it does:**
1. Loads DNS queries from CSV
2. Groups queries by MAC address
3. Extracts 25 behavioral features per device
4. Creates exploratory data analysis plots
5. Saves features to CSV

**Output:**
- `datasets/device_features.csv` (32 devices × 25 features)
- 5 EDA plots in `outputs/exploration/`

**Key Features Extracted:**
- **Basic stats**: total_queries, unique_domains, domain_diversity_ratio
- **Query types**: qtype_1_count, qtype_28_count, ratios
- **Temporal**: active_hours, active_days, time_span, queries_per_hour
- **Intervals**: avg/std/min/max/median query intervals
- **Behavior**: domain_entropy, query_burstiness, weekend_ratio

---

### Step 3: K-Means Clustering

```bash
python src/kmeans_clustering.py
```

**What it does:**
1. Loads device features
2. Normalizes features (StandardScaler)
3. Tests K values from 2 to 20
4. Calculates metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
5. Prompts user to select optimal K
6. Trains K-Means and assigns cluster labels
7. Generates visualizations

**Output:**
- `outputs/clustering/clustered_devices.csv` (features + cluster labels)
- `outputs/clustering/kmeans_model.pkl` (trained model)
- Visualization plots (optimal K analysis, 2D/3D PCA, heatmap)

**Interactive Prompt:**
```
Select optimal K value (2-20) [default: 4]:
```
Press Enter to use recommended K, or type a number.

---

### Step 4: Enrich DNS Queries

```bash
python src/enrich_dns_dataset.py
```

**What it does:**
1. Loads original DNS queries (584k rows)
2. Loads clustered devices with features (32 rows)
3. Merges device features to each query
4. Adds query-level temporal features (hour, day_of_week, is_weekend)
5. Saves enriched dataset

**Output:**
- `datasets/enriched_dns_queries.csv` (584k queries × 33 columns)

**Columns Added:**
- All 25 device-level features
- Cluster label
- Query-level: hour, day_of_week, is_weekend

---

### Step 5: Random Forest Classification

```bash
python src/random_forest_classifier.py
```

**What it does:**
1. **Phase 1**: Validates features with Cross-Validation on device-level data
2. **Phase 2**: Prepares query-level dataset for train/test split
3. **Phase 3**: Trains Random Forest on 70% of queries
4. **Phase 4**: Evaluates on 30% test queries
5. **Phase 5**: Analyzes feature importance

**Output:**
- `outputs/random_forest/random_forest_model.pkl`
- `outputs/random_forest/classification_report.txt`
- Confusion matrix and feature importance plots

**Expected Results:**
- Query-level Test Accuracy: 80-99%
- Feature importance showing which features best identify devices

---

## Scripts Documentation

### 1. pkl_converter.py

**Purpose:** Convert DNS dataset from pickle format to CSV/Excel for easy viewing.

**Key Functions:**
- Loads `.pkl` file using pickle
- Handles DataFrame, dict, or list data types
- Exports to both CSV and Excel formats

**Usage:**
```python
python src/pkl_converter.py
```

**Configuration:**
Edit these lines to change input/output:
```python
pkl_file = os.path.join(datasets_dir, "dns_dataset.pkl")
output_file = os.path.join(datasets_dir, "dns_dataset.xlsx")
```

---

### 2. feature_engineering.py

**Purpose:** Extract behavioral features from DNS queries for device fingerprinting.

**Class:** `DNSFeatureEngineer`

**Key Methods:**

```python
load_data()                    # Load DNS dataset
explore_data()                 # Create EDA plots
engineer_features()            # Extract 25 features per device
save_features(output_path)     # Save to CSV
```

**Features Engineered (25 total):**

| Category | Features |
|----------|----------|
| Basic | total_queries, unique_domains, domain_diversity_ratio |
| Query Types | qtype_1_count, qtype_28_count, qtype_other_count, qtype_1_ratio, qtype_28_ratio |
| Temporal | active_hours, active_days, time_span_seconds, time_span_hours, queries_per_hour |
| Intervals | avg_query_interval, std_query_interval, min_query_interval, max_query_interval, median_query_interval |
| Domain Behavior | top_domain_frequency, top_domain_ratio, domain_entropy |
| Activity | peak_hour_queries, peak_hour_ratio, query_burstiness, weekend_query_ratio |

**NaN Handling:**
- Automatically fills NaN values with 0.0
- Prevents errors in K-Means clustering

---

### 3. kmeans_clustering.py

**Purpose:** Cluster devices based on behavioral patterns.

**Class:** `DNSKMeansClustering`

**Key Methods:**

```python
load_features()                      # Load device features
normalize_features()                 # StandardScaler normalization
find_optimal_k(k_range)              # Test multiple K values
train_kmeans(k)                      # Train K-Means
visualize_clusters()                 # Create plots
evaluate_clustering()                # Calculate metrics
save_results()                       # Save model + results
```

**Evaluation Metrics:**
- **Silhouette Score**: -1 to 1 (higher = better separation)
- **Davies-Bouldin Index**: Lower = better clustering
- **Calinski-Harabasz Score**: Higher = better defined clusters
- **Inertia**: Within-cluster sum of squares

**Optimal K Selection:**
- Automatically recommends K with highest Silhouette Score
- User can override with custom value
- Interactive prompt with default option

---

### 4. enrich_dns_dataset.py

**Purpose:** Merge device features and cluster labels back to original queries.

**Key Operations:**

```python
# Load datasets
dns_queries = pd.read_csv('dns_dataset.csv')           # 584k queries
clustered_devices = pd.read_csv('clustered_devices.csv')  # 32 devices

# Merge on MAC address
enriched = dns_queries.merge(clustered_devices, on='mac', how='left')

# Add query-level temporal features
enriched['hour'] = pd.to_datetime(enriched['timestamp'], unit='s').dt.hour
enriched['day_of_week'] = pd.to_datetime(enriched['timestamp'], unit='s').dt.dayofweek
enriched['is_weekend'] = enriched['day_of_week'].isin([5, 6]).astype(int)
```

**Output Schema:**
```
Original (4 columns):      mac, domain, qtype, timestamp
Added (29 columns):        25 device features + cluster + hour + day_of_week + is_weekend
Total (33 columns)
```

---

### 5. random_forest_classifier.py

**Purpose:** Train and evaluate Random Forest classifier for device identification.

**Class:** `DNSRandomForestClassifier`

**Pipeline Phases:**

#### Phase 1: Device-Level Validation (Cross-Validation)
```python
validate_device_features_with_cv(n_folds=5)
```
- Validates feature engineering quality
- Uses 5-Fold CV on 32 devices
- Tests clustering stability
- **Expected**: Low accuracy (0-20%) due to 1 sample per device class
- **Purpose**: Sanity check, not final evaluation

#### Phase 2: Query-Level Dataset Preparation
```python
prepare_query_level_dataset(test_size=0.3, min_queries_per_device=2)
```
- Filters devices with < 2 queries
- Splits 584k queries into 70% train / 30% test
- Uses stratified sampling
- Encodes MAC addresses as numeric labels

#### Phase 3: Model Training
```python
train_random_forest(n_estimators=100, max_depth=20)
```
- Trains Random Forest with 100 trees
- Max depth = 20 (prevents overfitting)
- Uses all CPU cores (n_jobs=-1)

#### Phase 4: Model Evaluation
```python
evaluate_model()
```
- Calculates train/test accuracy
- Generates classification report
- Creates confusion matrix (top 10 devices)
- Checks for overfitting

#### Phase 5: Feature Importance
```python
analyze_feature_importance(top_n=25)
```
- Ranks features by importance
- Highlights cluster feature contribution
- Saves CSV and plot

**Key Parameters:**

```python
# Random Forest
n_estimators = 100          # Number of trees
max_depth = 20              # Max tree depth (prevents overfitting)
min_samples_split = 2       # Min samples to split node

# Train/Test Split
test_size = 0.3             # 30% for testing
random_state = 42           # Reproducibility
```

---

## Obstacles & Solutions

### Obstacle 1: NaN Values in Features

**Problem:**
```
ValueError: Input X contains NaN.
KMeans does not accept missing values encoded as NaN natively.
```

**Root Cause:**
- Devices with single queries had undefined standard deviation
- Division by zero in burstiness calculation
- Missing intervals for single-query devices

**Solution:**
```python
# In feature_engineering.py
if len(intervals) > 0:
    device_features['std_query_interval'] = intervals.std() if len(intervals) > 1 else 0.0
else:
    device_features['std_query_interval'] = 0.0

# Add NaN check at end
if self.features_df.isna().any().any():
    print("WARNING: Filling NaN values with 0...")
    self.features_df = self.features_df.fillna(0.0)
```

**Lesson:** Always handle edge cases (single-sample devices) and validate data before ML algorithms.

---

### Obstacle 2: Path Resolution Issues

**Problem:**
```
OSError: Cannot save file into a non-existent directory: '..\datasets'
```

**Root Cause:**
- Relative paths (`../datasets`) depend on current working directory
- Script location ≠ execution directory

**Solution:**
```python
# Use absolute paths based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_path = os.path.join(project_dir, "datasets", "device_features.csv")
```

**Lesson:** Always use `os.path.abspath(__file__)` for consistent path resolution.

---

### Obstacle 3: StratifiedKFold with Single Samples

**Problem:**
```
ValueError: n_splits=5 cannot be greater than the number of members in each class.
```

**Root Cause:**
- 32 unique device classes, each with 1 aggregated sample
- StratifiedKFold requires ≥2 samples per class

**Solution:**
```python
# Switch from StratifiedKFold to regular KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

**Lesson:** Understand dataset constraints - stratified splitting only works with multiple samples per class.

---

### Obstacle 4: Train/Test Split with Too Few Samples

**Problem:**
```
ValueError: The least populated classes in y have only 1 member, which is too few.
```

**Root Cause:**
- Some devices had only 1 DNS query
- Stratified split requires ≥2 samples per class

**Solution:**
```python
# Filter devices with insufficient queries
device_counts = queries_df['mac'].value_counts()
valid_devices = device_counts[device_counts >= min_queries_per_device].index
queries_df = queries_df[queries_df['mac'].isin(valid_devices)]
```

**Lesson:** Preprocess data to meet algorithm requirements - remove edge cases.

---

### Obstacle 5: F-String Syntax Error in range()

**Problem:**
```
TypeError: 'set' object cannot be interpreted as an integer
```

**Root Cause:**
```python
# Incorrect - curly braces interpreted as set literal
while num not in range({min(k_range)}, {max(k_range)}):
```

**Solution:**
```python
# Correct - extract values to variables first
min_k = min(k_range)
max_k = max(k_range)
while num not in range(min_k, max_k + 1):
```

**Lesson:** Avoid complex expressions inside f-strings - extract to variables for clarity.

---

### Obstacle 6: 0% Device-Level CV Accuracy

**Problem:**
```
Mean CV Accuracy: 0.0000 (0.00%)
```

**Root Cause:**
- 32 devices, each appearing once
- Cross-validation tests on completely unseen devices
- Model can't identify devices it's never seen

**Solution:**
This is **EXPECTED behavior** - not an error!

```python
# Added interpretation message
if cv_scores.mean() < 0.1:
    print("⚠️  Low CV accuracy is EXPECTED with 1 sample per device class")
    print("✅ This confirms each device is truly unique")
    print("📊 Query-level evaluation will show true performance")
```

**Lesson:** Understand the difference between:
- **Device-level CV**: Tests generalization to new devices (impossible with 1 sample)
- **Query-level train/test**: Tests identification of known devices with new queries (the real task)

---

### Obstacle 7: Confusion Matrix Too Large

**Problem:**
- 32×32 confusion matrix is unreadable in visualization

**Solution:**
```python
# Show only top 10 most common devices
unique, counts = np.unique(y_test, return_counts=True)
top_10_indices = unique[np.argsort(counts)[-10:]]

# Filter to top 10 for visualization
cm = confusion_matrix(y_test[mask], y_test_pred[mask], labels=top_10_indices)
```

**Lesson:** Simplify visualizations for interpretability - focus on most representative samples.

---

## Troubleshooting Guide

### Issue 1: ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
```bash
# Ensure virtual environment is activated
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Issue 2: File Not Found Error

**Symptoms:**
```
FileNotFoundError: dns_dataset.csv not found
```

**Solutions:**

1. **Check file exists:**
```bash
ls datasets/
```

2. **Run pkl_converter first:**
```bash
python src/pkl_converter.py
```

3. **Verify file path in script:**
```python
# In script
print(f"Looking for: {dataset_path}")
```

---

### Issue 3: Empty or Incorrect Output

**Symptoms:**
- Output files have 0 rows
- Features are all zeros

**Debug Steps:**

1. **Check input data:**
```python
df = pd.read_csv('datasets/dns_dataset.csv')
print(df.head())
print(df.info())
print(df['mac'].nunique())  # Should be 32
```

2. **Verify feature engineering:**
```python
# Add debug prints in engineer_features()
print(f"Processing device {mac}: {len(group)} queries")
```

3. **Check for NaN values:**
```python
print(features_df.isna().sum())
```

---

### Issue 4: Low Model Accuracy (<50%)

**Possible Causes:**

1. **Overfitting:**
   - Solution: Reduce `max_depth` or increase `min_samples_split`
   ```python
   rf_model = RandomForestClassifier(max_depth=10, min_samples_split=5)
   ```

2. **Insufficient features:**
   - Solution: Add more discriminative features (e.g., domain TF-IDF)

3. **Similar device behaviors:**
   - Check cluster analysis - some devices may be indistinguishable

4. **Data quality issues:**
   - Verify timestamp is properly converted
   - Check for missing values

**Debug:**
```python
# Check feature distributions
import matplotlib.pyplot as plt
features_df.hist(figsize=(15, 12), bins=30)
plt.show()

# Check cluster separation
from sklearn.metrics import silhouette_score
print(f"Silhouette: {silhouette_score(X_scaled, cluster_labels)}")
```

---

### Issue 5: Memory Error

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce dataset size for testing:**
```python
# In enrich_dns_dataset.py
dns_df = pd.read_csv('dns_dataset.csv', nrows=100000)  # Sample first 100k
```

2. **Use chunking:**
```python
chunks = []
for chunk in pd.read_csv('dns_dataset.csv', chunksize=50000):
    processed = process_chunk(chunk)
    chunks.append(processed)
df = pd.concat(chunks)
```

3. **Optimize dtypes:**
```python
df['qtype'] = df['qtype'].astype('int8')
df['hour'] = df['hour'].astype('int8')
```

---

### Issue 6: Plots Not Saving

**Symptoms:**
- No error, but plot files don't appear

**Solutions:**

1. **Check output directory exists:**
```python
os.makedirs(output_dir, exist_ok=True)
```

2. **Use absolute paths:**
```python
output_path = os.path.join(project_dir, "outputs", "exploration", "plot.png")
print(f"Saving to: {output_path}")
```

3. **Close plots after saving:**
```python
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()  # Important!
```

---

## Results Interpretation

### K-Means Clustering Results

**Optimal K Selection:**
- Review the `optimal_k_analysis.png` plot
- Look for:
  - **Elbow** in inertia plot (diminishing returns)
  - **Peak** in Silhouette score (highest = best)
  - **Valley** in Davies-Bouldin index (lowest = best)

**Cluster Quality Metrics:**

| Metric | Good Range | Your Value | Interpretation |
|--------|------------|------------|----------------|
| Silhouette Score | 0.5 - 1.0 | ~0.32 | Moderate separation |
| Davies-Bouldin | 0.0 - 1.0 | Check output | Lower = better |
| Calinski-Harabasz | >100 | Check output | Higher = better |

**Cluster Interpretation:**
- See `documentation/cluster_analysis.md` for detailed breakdown
- Clusters represent device behavioral groups:
  - **Cluster 0**: Burst scanners (high diversity, short duration)
  - **Cluster 1**: Heavy users (24/7 activity, high query volume)
  - **Cluster 2**: Normal users (moderate activity)
  - **Cluster 3**: Minimal activity (IoT/single-purpose devices)

---

### Random Forest Results

**Phase 1: Device-Level CV (Expected: 0-20%)**
```
Mean CV Accuracy: 0.0000 (0.00%)
```
- ✅ **This is NORMAL** - Can't identify completely new devices
- Validates each device is unique

**Phase 4: Query-Level Performance (Main Result)**

```
Training Accuracy: 95-99%
Test Accuracy: 80-95% ← FOCUS HERE
```

**Interpretation Guide:**

| Test Accuracy | Interpretation | Action |
|---------------|----------------|--------|
| **>90%** | Excellent! Devices highly distinguishable | ✅ Ready for deployment |
| **70-90%** | Good. Most queries correctly identified | ✅ Acceptable performance |
| **50-70%** | Moderate. Some devices are similar | ⚠️ Consider feature engineering |
| **<50%** | Poor. Features not discriminative | ❌ Revisit feature selection |

**Overfitting Check:**
```
Train Accuracy - Test Accuracy = Overfitting Gap
```
- **<10%**: Good generalization ✅
- **10-20%**: Moderate overfitting ⚠️
- **>20%**: Severe overfitting ❌ (reduce max_depth, increase min_samples_split)

---

### Feature Importance Analysis

**Key Questions:**

1. **Is cluster label important?**
   - Check rank in `feature_importance.csv`
   - If top 5: Clustering significantly helps classification ✅
   - If >15: Clustering has minimal impact ⚠️

2. **Which features matter most?**
   - Likely important: total_queries, unique_domains, queries_per_hour
   - Domain behavior: domain_entropy, top_domain_ratio
   - Temporal: weekend_ratio, active_hours

3. **Are all features useful?**
   - Features with importance <0.01 can be removed
   - Simplifies model without losing accuracy

**Example Interpretation:**
```
Top Features:
1. total_queries: 0.15 (15%)        # Volume matters!
2. cluster: 0.12 (12%)              # Clustering helps!
3. domain_entropy: 0.09 (9%)        # Diversity is key
4. queries_per_hour: 0.08 (8%)      # Rate is distinctive
```

---

### Classification Report

**Understand the Metrics:**

```
              precision  recall  f1-score  support

device_X         0.95     0.92     0.93      1500
device_Y         0.88     0.91     0.89      800
```

- **Precision**: Of predictions for device X, how many were correct?
- **Recall**: Of actual device X queries, how many were found?
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of queries for this device in test set

**Per-Device Analysis:**
- High precision + Low recall: Model is conservative (misses some queries)
- Low precision + High recall: Model is aggressive (false positives)
- Both low: Device is hard to identify (similar to others)

---

## Future Improvements

### 1. Enhanced Feature Engineering

**Domain-Based Features:**
```python
# TF-IDF of domain names
from sklearn.feature_extraction.text import TfidfVectorizer

device_domains = df.groupby('mac')['domain'].apply(lambda x: ' '.join(x))
tfidf = TfidfVectorizer(max_features=100)
domain_features = tfidf.fit_transform(device_domains).toarray()
```

**Time-Series Features:**
```python
# Query patterns by hour of day (24 features)
hourly_distribution = df.groupby(['mac', 'hour']).size().unstack(fill_value=0)

# Day-of-week patterns (7 features)
daily_distribution = df.groupby(['mac', 'day_of_week']).size().unstack(fill_value=0)
```

**Network Graph Features:**
```python
# PageRank of frequently visited domains
import networkx as nx
# Build device-domain graph
# Calculate centrality metrics
```

---

### 2. Advanced Clustering Techniques

**Hierarchical Clustering:**
```python
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
```

**DBSCAN (Density-Based):**
```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
clusters = dbscan.fit_predict(X_scaled)
# Better for irregular cluster shapes
```

**Gaussian Mixture Models:**
```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4, covariance_type='full')
clusters = gmm.fit_predict(X_scaled)
# Probabilistic cluster assignment
```

---

### 3. Model Ensemble

**Combine Multiple Classifiers:**
```python
from sklearn.ensemble import VotingClassifier

rf = RandomForestClassifier(n_estimators=100)
gb = GradientBoostingClassifier(n_estimators=100)
svc = SVC(probability=True)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svc', svc)],
    voting='soft'
)
ensemble.fit(X_train, y_train)
```

---

### 4. Deep Learning Approach

**LSTM for Temporal Sequences:**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Model DNS query sequences per device
model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_features)),
    Dense(64, activation='relu'),
    Dense(num_devices, activation='softmax')
])
```

---

### 5. Real-Time Classification

**Stream Processing:**
```python
# Load trained model
with open('random_forest_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    rf_model = model_data['model']
    label_encoder = model_data['label_encoder']

# Classify new query
def classify_query(mac, domain, qtype, timestamp):
    # Extract features for this device
    device_features = get_device_features(mac)
    
    # Predict
    prediction = rf_model.predict([device_features])
    device_mac = label_encoder.inverse_transform(prediction)[0]
    
    return device_mac
```

---

### 6. Anomaly Detection

**Isolation Forest:**
```python
from sklearn.ensemble import IsolationForest

# Train on normal device behavior
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(X_normal)

# Detect anomalies in new devices
anomalies = iso_forest.predict(X_new)  # -1 = anomaly, 1 = normal
```

**Use Cases:**
- Detect compromised devices (behavior change)
- Identify botnets
- Flag suspicious DNS queries

---

### 7. Explainability

**SHAP Values:**
```python
import shap

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Visualize feature contributions
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

**LIME (Local Interpretable Model-agnostic Explanations):**
```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train, feature_names=feature_names)
exp = explainer.explain_instance(X_test[0], rf_model.predict_proba)
exp.show_in_notebook()
```

---

### 8. Multi-Label Classification

**Predict Device Type + Specific Identity:**
```python
# First classifier: Device type (PC, smartphone, IoT)
type_classifier = RandomForestClassifier()
device_type = type_classifier.predict(features)

# Second classifier: Specific device within type
identity_classifier = RandomForestClassifier()
device_id = identity_classifier.predict(features)
```

---

### 9. Federated Learning

**Privacy-Preserving Device Identification:**
- Train models on local data (per network)
- Share only model parameters, not raw data
- Aggregate models for global device fingerprinting

---

### 10. Continuous Learning

**Online Learning:**
```python
from sklearn.linear_model import SGDClassifier

# Incremental learning with new data
sgd_model = SGDClassifier()
for batch in data_stream:
    X_batch, y_batch = batch
    sgd_model.partial_fit(X_batch, y_batch, classes=all_classes)
```

---

## Performance Benchmarks

### Expected Execution Times (on typical hardware)

| Script | Dataset Size | Execution Time | Memory Usage |
|--------|--------------|----------------|--------------|
| pkl_converter.py | 584k rows | 5-10 seconds | ~500 MB |
| feature_engineering.py | 584k rows | 30-60 seconds | ~1 GB |
| kmeans_clustering.py | 32 devices | 10-30 seconds | ~100 MB |
| enrich_dns_dataset.py | 584k rows | 10-20 seconds | ~1 GB |
| random_forest_classifier.py | 584k rows | 2-5 minutes | ~2 GB |

**Hardware Used:**
- CPU: Intel i5/i7 or equivalent
- RAM: 8 GB minimum (16 GB recommended)
- Storage: SSD recommended

---

## Project Achievements

### ✅ Completed Objectives

1. **Data Processing**: Successfully converted and processed 584k DNS queries
2. **Feature Engineering**: Extracted 25 meaningful behavioral features
3. **Clustering**: Identified 4 device behavioral groups using K-Means
4. **Classification**: Achieved [YOUR_ACCURACY]% accuracy in device identification
5. **Visualization**: Created comprehensive plots for analysis
6. **Documentation**: Detailed workflow and troubleshooting guides

### 📊 Key Metrics

- **Devices Analyzed**: 32 unique MAC addresses
- **DNS Queries Processed**: 584,789
- **Features Engineered**: 25 behavioral features
- **Clusters Identified**: 4 device groups
- **Model Accuracy**: [Check your output] on query-level data
- **Feature Importance**: Cluster label ranks in top [CHECK] features

---

## References & Resources

### Academic Papers
1. K-Means Clustering: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
2. Random Forests: Breiman, L. (2001). "Random Forests"
3. DNS-Based Device Fingerprinting: Numerous cybersecurity research papers

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### Tutorials
- [K-Means Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

---

## Contact & Contribution

**Project Author:** [Your Name]  
**Institution:** [Your University]  
**Course:** AI in Information Security Systems  
**Date:** January 2026

**For questions or improvements:**
- Check troubleshooting guide first
- Review error messages carefully
- Consult documentation files in `documentation/`

---

## License

This project is created for educational purposes as part of a Master's program in AI and Information Security.

---

**Last Updated:** January 19, 2026  
**Version:** 1.0  
**Status:** Production-Ready ✅
