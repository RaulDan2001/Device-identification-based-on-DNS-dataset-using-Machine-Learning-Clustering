┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Feature Engineering (Device-Level)                 │
│ - Input: 584k DNS queries                                   │
│ - Group by MAC → 32 devices                                 │
│ - Engineer 25 features per device                           │
│ - Validate with Cross-Validation                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: K-Means Clustering (Device-Level)                  │
│ - Input: 32 devices with 25 features                        │
│ - Find optimal K, assign cluster labels                     │
│ - Validate with Silhouette Score, CV                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Enrich Original Dataset (Query-Level) ← NEW!       │
│ - Merge cluster labels back to 584k queries                 │
│ - Add device features to each query                         │
│ - Create query-level features (hour, day_of_week)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Random Forest Classification (Query-Level)         │
│ - Input: 584k enriched queries                              │
│ - Train/Test Split (70/30) = 408k / 175k                    │
│ - Predict MAC from query + device context                   │
│ - High confidence with large test set!                      │
└─────────────────────────────────────────────────────────────┘