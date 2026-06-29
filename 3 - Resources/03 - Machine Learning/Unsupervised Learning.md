---
tags: [machine-learning, unsupervised, core]
status: growing
created: 2026-06-27
---

# Unsupervised Learning

## 1. Escenario de aprendizaje

Trabajas en una empresa de comercio electrónico con millones de clientes. Tienes datos de compras, navegación y soporte técnico, pero no tienes etiquetas — nadie te ha dicho "este cliente es de tipo A, este es de tipo B". Necesitas encontrar segmentos de clientes por su comportamiento para personalizar ofertas y campañas de marketing. Sin etiquetas disponibles, el unsupervised learning descubre **estructura oculta en los datos**: agrupa clientes similares, reduce dimensionalidad para visualización y detecta anomalías como fraude.

Tres grandes tareas no supervisadas:
- **Clustering**: agrupar ítems similares
- **Dimensionality reduction**: comprimir datos preservando estructura
- **Anomaly detection**: encontrar patrones inusuales

Estas son esenciales para el análisis exploratorio de datos, feature engineering y entender tus datos antes de aplicar métodos supervisados.

---

## 2. Clustering

### 2.1 K-Means

**Goal**: partition $n$ points into $k$ clusters, minimizing within-cluster variance.

**The algorithm** (step by step):
1. Pick $k$ random points as initial centroids
2. **Assign**: each point goes to the nearest centroid
3. **Update**: recompute centroids as the mean of points in each cluster
4. Repeat steps 2-3 until convergence (centroids stop changing)

**Concrete example** — clustering customers:
```
Customers: (age, income) = [(25, 30k), (30, 35k), (55, 80k), (60, 85k)]
k = 2

Step 1: centroid1 = (25, 30k), centroid2 = (55, 80k)  (random init)
Step 2: customer (30, 35k) → centroid1 (closer)
         customer (60, 85k) → centroid2 (closer)
Step 3: centroid1 = ((25+30)/2, (30+35)/2) = (27.5, 32.5k)
         centroid2 = ((55+60)/2, (80+85)/2) = (57.5, 82.5k)
Repeat → converges in a few iterations
```

**Choosing $k$**:
- **Elbow method**: plot inertia (within-cluster variance) vs $k$. Look for the "elbow" where adding more clusters gives diminishing returns.
- **Silhouette score**: measures how similar points are to their own cluster vs other clusters. Higher is better.

**Limitations**:
- Need to specify $k$ upfront
- Sensitive to initialization (k-means++ helps)
- Assumes spherical clusters of similar size
- Struggles with non-convex shapes, varying densities

### 2.2 DBSCAN

**Density-based clustering**: clusters are regions of high density separated by regions of low density.

**How it works**:
1. For each point, count neighbors within radius $\epsilon$
2. If neighbors $\geq$ minPts, the point is a **core point** (start a cluster)
3. All points within $\epsilon$ of a core point belong to the same cluster
4. Points not reachable from any core point are **noise** (outliers)

**Advantages over K-Means**:
- Does not require $k$
- Can find arbitrarily shaped clusters
- Automatically detects outliers
- Robust to noise

**Disadvantages**:
- Sensitive to $\epsilon$ and minPts parameters
- Struggles with varying densities
- Does not work well in high dimensions (curse of dimensionality)

### 2.3 Hierarchical Clustering

Builds a tree (dendrogram) of clusters.

**Agglomerative** (bottom-up, most common):
1. Start: each point is its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains

**Divisive** (top-down): start with one cluster, split recursively.

The dendrogram lets you choose the number of clusters after the fact — just cut the tree at any level.

---

## 3. Dimensionality Reduction

### 3.1 Principal Component Analysis (PCA)

**Goal**: find a lower-dimensional representation that preserves maximum variance.

**Intuition**: imagine a cloud of points in 3D shaped like a flat pancake. Most of the variance (information) is in the plane of the pancake; the thickness is noise. PCA finds the plane.

**How it works** (via SVD):
1. Center the data: subtract the mean of each feature
2. Compute $X = U \Sigma V^T$ (SVD of centered data)
3. Principal components = columns of $V$ (directions of maximum variance; see [[Linear Algebra]] for SVD details)
4. Projections = $U_k \Sigma_k$ (data in reduced space)

**Choosing the number of components**:
- Look at explained variance ratio: $\frac{\sigma_i^2}{\sum \sigma_j^2}$ (see [[Statistics]] for variance concepts)
- Choose $k$ such that cumulative variance > 0.9 or 0.95

**When to use PCA**:
- Visualization (project to 2D or 3D)
- Noise reduction (small components are often noise)
- Speed up training (fewer features)
- Before distance-based algorithms (KNN, K-Means) in high dimensions

**Warning**: PCA assumes linear structure. It does not capture non-linear relationships.

### 3.2 t-SNE

**Goal**: visualize high-dimensional data in 2D or 3D while preserving local structure.

**Why it is different from PCA**:
- PCA preserves global variance (large-scale structure)
- t-SNE preserves local neighborhoods (nearby points stay nearby)

t-SNE is non-deterministic, expensive (O(n²)), and should be used **only for visualization**, not for feature extraction or downstream modeling.

### 3.3 UMAP

Faster than t-SNE, better at preserving global structure, and scalable to larger datasets. Increasingly the default choice for embedding visualization.

---

## 4. Practical Considerations

| Task | Algorithm | When to Use |
|---|---|---|
| **Clustering** | K-Means | Large data, spherical clusters, known $k$ |
| | DBSCAN | Arbitrary shapes, unknown $k$, outlier detection |
| | Hierarchical | Small data, want dendrogram, interpretability |
| **Dim. reduction** | PCA | Linear structure, speed, preprocessing before ML |
| | t-SNE | Visualization only, small to medium data |
| | UMAP | Visualization, larger data, structure preservation |
| **Anomaly** | Isolation Forest | High dimensions, large data |
| | LOF | Local density anomalies |
| | Autoencoder | Complex patterns, enough data |

---

## 5. Common Mistakes

1. **PCA without scaling**: if features are on different scales, PCA directions are dominated by high-variance features. Always standardize first.

2. **Using t-SNE for feature extraction**: t-SNE is non-deterministic and distances are not meaningful. It is a visualization tool, not a preprocessing step.

3. **Assuming clusters found by K-Means are "real"**: K-Means always finds $k$ clusters, even in uniformly random data (see [[Probability]]). Validate with silhouette score or domain knowledge.

4. **Not checking for outliers before K-Means**: centroids can be dragged by outliers. Remove or clip extreme values.

5. **Interpreting PCA components as "features"**: each component is a linear combination of original features. They are not necessarily interpretable.

---

## 6. Check Your Understanding

1. You run K-Means with k=3 on random uniform data. What will the clusters look like? What does this tell you about evaluating clustering?
2. Why does PCA require scaling the data first? What happens if you skip it?
3. DBSCAN labels some points as "noise" (-1). What does this mean and why is it useful?
4. You have 500 features and want to visualize your data. Why might PCA be insufficient and t-SNE be a better choice?
5. How is the explained variance ratio related to eigenvalues in PCA?

---

## 7. Resumen

Unsupervised learning finds hidden structure in unlabeled data. Clustering groups similar points (K-Means for spherical clusters, DBSCAN for arbitrary shapes). Dimensionality reduction compresses data while preserving information (PCA for linear structure, t-SNE/UMAP for visualization). These methods are essential for understanding data before applying supervised learning, and for feature engineering in high-dimensional problems.

---

## 8. Where to Go Next

- [[Supervised Learning]] — Using representations found by unsupervised methods
- [[Feature Engineering]] — PCA as a feature extraction technique
- [[Model Evaluation]] — Evaluating clustering quality
