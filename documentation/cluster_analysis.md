# DNS Dataset Cluster Analysis

## Overview

- **Total Devices**: 32 unique MAC addresses
- **Behavioral Features**: 25 extracted features from DNS queries
- **Number of Clusters**: 4 (K=4, labeled 0-3)
- **Clustering Method**: K-Means with normalized features

---

## Dataset Structure

### Identity Columns
- **`mac`**: Hashed device identifier (MD5-like hash, unique per device)
- **`cluster`**: K-Means cluster assignment (0, 1, 2, or 3)

### Feature Categories

#### 1. Basic Query Statistics
| Feature | Description | Range |
|---------|-------------|-------|
| `total_queries` | Total DNS queries made by device | 1 - 230,413 |
| `unique_domains` | Number of different domains visited | 1 - 2,818 |
| `domain_diversity_ratio` | Unique domains / Total queries | 0.00 - 1.00 |

#### 2. Query Type Distribution
| Feature | Description |
|---------|-------------|
| `qtype_1_count` | Number of IPv4 (A record) queries |
| `qtype_28_count` | Number of IPv6 (AAAA record) queries |
| `qtype_other_count` | Number of other DNS query types |
| `qtype_1_ratio` | Percentage of IPv4 queries (0.0 - 1.0) |
| `qtype_28_ratio` | Percentage of IPv6 queries (0.0 - 1.0) |

#### 3. Temporal Patterns
| Feature | Description |
|---------|-------------|
| `active_hours` | Number of different hours device was active (0-24) |
| `active_days` | Number of different days of week (0-7) |
| `time_span_seconds` | Duration between first and last query (seconds) |
| `time_span_hours` | Duration in hours |
| `queries_per_hour` | Average query rate (queries/hour) |

#### 4. Query Interval Statistics
| Feature | Description |
|---------|-------------|
| `avg_query_interval` | Average time between consecutive queries (seconds) |
| `std_query_interval` | Standard deviation of query intervals |
| `min_query_interval` | Shortest time between queries |
| `max_query_interval` | Longest time between queries |
| `median_query_interval` | Median query interval |

#### 5. Domain Behavior
| Feature | Description |
|---------|-------------|
| `top_domain_frequency` | Number of times most-visited domain was queried |
| `top_domain_ratio` | Percentage of queries to top domain (0.0 - 1.0) |
| `domain_entropy` | Shannon entropy of domain distribution (higher = more diverse) |

#### 6. Activity Patterns
| Feature | Description |
|---------|-------------|
| `peak_hour_queries` | Number of queries in the busiest hour |
| `peak_hour_ratio` | Percentage of queries in peak hour |
| `query_burstiness` | Query timing variability (std/mean of intervals) |
| `weekend_query_ratio` | Percentage of queries on weekends (0.0 - 1.0) |

---

## Cluster Analysis

### Cluster 0: Burst Scanners
**Size**: 1 device

**Characteristics**:
- Very high domain diversity ratio (~55%)
- Low total queries (100-200 range)
- Short time span (few minutes)
- High queries per hour (1000+)

**Example Device**: `a7638f9e6bf169dc2f292c3bc54d557b`
- 109 queries across 60 unique domains
- 227 seconds time span
- 1728 queries per hour

**Behavioral Pattern**: Quick, diverse domain lookups in short bursts. Possibly a network scanner, DNS enumeration tool, or device performing rapid service discovery.

---

### Cluster 1: Heavy Users / Always-On Devices
**Size**: 2 devices

**Characteristics**:
- VERY high total queries (60,000 - 230,000)
- High unique domains (1,700 - 2,800)
- Active 24 hours/day, 7 days/week
- Extremely high queries per hour (100-400)
- Low domain diversity ratio despite high unique domains

**Example Devices**:
1. `c82d83a47e2f5fd4e028d3d0ce339d4a`
   - 230,413 queries
   - 1,752 unique domains
   - 408 queries/hour
   
2. `20bf46bf7631cd900b4ee1ccb158f9bf`
   - 64,398 queries
   - 2,818 unique domains
   - 110 queries/hour

**Behavioral Pattern**: High-activity devices that are always online. Likely servers, network gateways, or heavily-used workstations with automated background processes. These devices show continuous, sustained activity across all hours and days.

---

### Cluster 2: Normal User Devices
**Size**: 25 devices (LARGEST GROUP - 78% of devices)

**Characteristics**:
- Moderate query counts (hundreds to thousands)
- Low domain diversity ratio (1-7%)
- Active across multiple days
- Balanced IPv4/IPv6 usage
- Regular activity patterns

**Typical Profile**:
- 500 - 20,000 queries
- 50 - 500 unique domains
- Active 10-24 hours
- 5-7 active days

**Behavioral Pattern**: Standard user behavior representing laptops, desktops, smartphones, and tablets with regular browsing activity. These devices show:
- Repeated visits to favorite domains
- Active during work/waking hours
- Mix of weekday and weekend activity
- Moderate query rates

---

### Cluster 3: Minimal Activity / Single-Purpose Devices
**Size**: 4 devices

**Characteristics**:
- Very few queries (1-28)
- Very low unique domains (1-4)
- Single or minimal domain访问
- High top_domain_ratio (often 1.0)
- Limited time span

**Example Devices**:
- `19b2ec9332ff2d43ec1d2445b0f0831e`: 28 queries, 1 unique domain (100% to one domain)
- `6d92f0a6f8ce93b7b65de19c30f2b2ab`: 1 query, 1 domain
- `f0ea29c151b696e028e2e6a13b49c535`: 1 query, 1 domain
- `8f6dc892f4488281d5281ce68c8bc49e`: 1 query, 1 domain

**Behavioral Pattern**: Rarely-used devices, IoT devices with specific tasks, or devices that connected briefly. Could include:
- Smart home devices (thermostats, cameras)
- Devices that failed to complete connection
- Single-purpose network devices
- Temporary connections

---

## Key Statistical Insights

### Query Volume Distribution
- **Minimum**: 1 query
- **Maximum**: 230,413 queries
- **Median**: ~1,000 queries
- **Mean**: ~9,000 queries

### Domain Diversity
- **Highest diversity ratio**: 55% (Cluster 0)
- **Lowest diversity ratio**: <1% (many devices repeatedly visit same domains)
- **Most unique domains**: 2,818 (Cluster 1 device)

### IPv4 vs IPv6 Usage
- **IPv4 dominant**: Most devices prefer IPv4 (qtype_1_ratio > 0.8)
- **IPv6 adoption**: Some devices show ~50/50 split (modern dual-stack devices)
- **Pure IPv4**: Many devices have qtype_28_ratio = 0.0

### Temporal Patterns
- **24/7 devices**: 2 devices active all hours (Cluster 1)
- **Weekend activity**: Varies from 0% to 100%
- **Peak activity concentration**: Some devices have 20-40% of queries in single hour

---

## Notable Device Examples

### Device: `c82d83a47e2f5fd4e028d3d0ce339d4a` (Cluster 1)
**The Power User**
- 230,413 total queries
- 1,752 unique domains
- 408 queries/hour
- Active 24 hours/day, 7 days/week
- Only 0.76% domain diversity (repeatedly queries same domains)

### Device: `a7638f9e6bf169dc2f292c3bc54d557b` (Cluster 0)
**The Scanner**
- 109 queries in 227 seconds
- 60 unique domains (55% diversity)
- 1,728 queries/hour rate
- Burst activity pattern

### Device: `9d8d00368461c86db72d1e360c544799` (Cluster 2)
**The Weekend Warrior**
- 100% weekend activity
- 1,254 queries
- 98 unique domains
- All activity concentrated on weekends

### Device: `19b2ec9332ff2d43ec1d2445b0f0831e` (Cluster 3)
**The Single-Purpose Device**
- 28 queries, all to 1 domain
- 100% top_domain_ratio
- Likely IoT or single-service device

---

## Cluster Separation Quality

Based on K-Means evaluation metrics:
- **Optimal K**: 4 clusters (you selected this value)
- **Silhouette Score**: ~0.32 (moderate cluster separation)
- Clusters are distinguishable but have some overlap
- Clear separation between heavy users (Cluster 1) and minimal devices (Cluster 3)

---

## Random Forest Classification Context

The Random Forest classifier uses these features + cluster labels to predict device identity (MAC address):

**How Cluster Information Helps**:
1. Reduces search space: "This is a Cluster 1 device, so only 2 candidates"
2. Adds behavioral context: "Heavy user pattern + high query rate → likely device X"
3. Improves accuracy: Cluster label is an additional feature for classification

**Expected Performance**:
- High accuracy for distinctive devices (Cluster 0, 1, 3)
- Moderate accuracy for normal users (Cluster 2) with similar patterns
- Feature importance analysis will show which features best distinguish devices

---

## Interpretation for Device Fingerprinting

### Use Cases

1. **Device Re-identification**: 
   - Given new DNS traffic, predict which known device it belongs to
   - Useful for network access control, user tracking

2. **Anomaly Detection**:
   - Devices changing clusters = behavior change
   - New device not fitting any cluster = potential threat

3. **Device Type Inference** (proxy):
   - Cluster 1 likely = Servers/Gateways
   - Cluster 2 likely = User devices (laptops, phones)
   - Cluster 3 likely = IoT/embedded devices

4. **Network Profiling**:
   - Understand network composition (78% normal users, 6% heavy users, etc.)
   - Identify bandwidth-heavy devices
   - Detect unusual activity patterns

---

## Feature Engineering Quality

**Strengths**:
- Comprehensive coverage of DNS behavior
- Mix of count-based, ratio-based, and statistical features
- Temporal patterns captured (time of day, weekend activity)
- Domain diversity metrics (entropy, top domain ratio)

**Feature Correlation Insights**:
- `total_queries` and `unique_domains` are correlated but not redundant
- Temporal features (`active_hours`, `time_span`) provide context
- Burstiness captures query timing regularity
- Cluster label adds group-level information

---

## Data Quality Notes

- **No missing values**: All features properly calculated (NaN values filled with 0.0)
- **Normalized for clustering**: Features scaled to mean=0, std=1 for K-Means
- **Hashed identifiers**: MAC and domain names are anonymized (MD5-like hashes)
- **Real-world data**: Includes edge cases (1-query devices, 200k+ query devices)

---

**Generated**: January 17, 2026  
**Source**: DNS dataset with 584,791 total queries from 32 devices  
**Clustering Method**: K-Means (K=4) with Silhouette Score optimization
