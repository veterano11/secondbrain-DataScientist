---
tags: [mathematics, linear-algebra, foundational]
status: growing
created: 2026-06-27
---

# Linear Algebra

## 1. Why This Matters

Every time you train an ML model, you are doing linear algebra. When you ask ChatGPT to generate text, every token is a vector and the attention mechanism is a series of matrix multiplications. When you run PCA, you are decomposing a covariance matrix into its fundamental components. Linear algebra **is the language in which machine learning is written**. Without it, models are black boxes. With it, you see that everything is vectors being transformed.

In this note we build from simple vectors to the most powerful decompositions, always connecting to how they are used in ML.

---

## 2. Vectors

### 2.1 Intuition

A vector is an **arrow in space** with a direction and a magnitude. But it is also an **ordered list of numbers**. Both views are the same.

Imagine a point on a 2D Cartesian plane. The vector $v = (3, 4)$ means "walk 3 units right and 4 up". That is a vector.

Every row of your dataset $X$ is a vector. If your dataset has 100 people with 5 measurements (age, height, income, education, hours_slept), each person is a vector in $\mathbb{R}^5$. You cannot visualize 5 dimensions, but mathematically it works exactly like 2 or 3.

### 2.2 Formal Definition and Operations

A vector $v \in \mathbb{R}^n$ is an ordered tuple of $n$ real numbers.

**Addition**: $v + w = (v_1 + w_1, v_2 + w_2, ...)$ — component by component. Place one arrow at the tip of the other.

**Scalar multiplication**: $c \cdot v = (c \cdot v_1, c \cdot v_2, ...)$ — stretch or shrink the arrow. If $c < 0$, reverse direction.

### 2.3 The Dot Product

This is where the magic begins.

$$v \cdot w = \sum_{i=1}^n v_i w_i = v_1 w_1 + v_2 w_2 + ... + v_n w_n$$

Equivalently:

$$v \cdot w = \|v\| \cdot \|w\| \cdot \cos\theta$$

where $\theta$ is the angle between the vectors.

**Concrete example**:
```
v = [3, 4]
w = [1, 2]

v · w = 3×1 + 4×2 = 3 + 8 = 11
‖v‖ = √(3² + 4²) = √25 = 5
‖w‖ = √(1² + 2²) = √5 ≈ 2.236

cos θ = 11 / (5 × 2.236) = 11 / 11.18 ≈ 0.984
θ ≈ 10.3°
```

**What does the dot product tell us?**

- **Measures similarity**: if two vectors point in the same direction, the dot product is large and positive. Opposite → negative. Perpendicular → **zero**.
- **It is the heart of attention in transformers**: $QK^T$ is a matrix of dot products between queries and keys. Each cell $(i, j)$ says "how much does token $i$ relate to token $j$".
- **It is the basis of cosine similarity**: $\text{cosine sim}(v, w) = \frac{v \cdot w}{\|v\| \|w\|}$, which is what you use in RAG to find similar documents.

### 2.4 Norm (Magnitude)

The L2 norm (the most common) is the length of the arrow:

$$\|v\|_2 = \sqrt{\sum v_i^2}$$

For $v = [3, 4]$: $\|v\| = \sqrt{9 + 16} = 5$. This is the hypotenuse of a 3-4-5 triangle.

**L1 norm**: $\|v\|_1 = \sum |v_i| = 3 + 4 = 7$. Used in Lasso regularization — it drives weights to exactly zero.

**Why normalize?** If one feature is "age" (0-100) and another is "income" (0-1,000,000), the dot product will be dominated by income. Normalizing (unit vectors where $\|v\| = 1$) ensures all features weigh equally.

### 2.5 Linear Independence

A set of vectors is **linearly independent** if no single vector can be written as a combination of the others.

```
v₁ = [1, 0]
v₂ = [0, 1]    → Independent (they form a basis)
v₃ = [2, 3]    → Dependent on v₁ and v₂

Because v₃ = 2·v₁ + 3·v₂
```

**Why does this matter?** In ML, linear independence relates to **multicollinearity**. If two features are linearly dependent (or nearly so), your model will have unstable coefficients. That is why we do feature selection and regularization.

---

## 3. Matrices

### 3.1 Intuition

A matrix is a **collection of vectors** organized in rows and columns. But it is also a **transformation**: multiplying a matrix by a vector rotates, scales, and reflects that vector.

Your dataset $X$ is an $n \times d$ matrix: $n$ rows (samples), $d$ columns (features).

### 3.2 Matrix-Vector Multiplication

$$A \cdot v = w$$

Each element of $w$ is the dot product of a row of $A$ with $v$:

```
A = [1 2]    v = [3]    A·v = [1×3 + 2×4] = [11]
    [3 4]        [4]          [3×3 + 4×4]   [25]
```

**Visualization**: the matrix transforms space. If you take every point in a square and multiply by a matrix, you get a rotated, stretched parallelogram.

**In a neural network**: each layer does $h = Wx + b$. $W$ is a matrix that transforms the input vector $x$ into the hidden vector $h$. Learning is finding the right $W$.

### 3.3 Matrix-Matrix Multiplication

$$C = A \cdot B$$

Each column of $C$ is $A$ times the corresponding column of $B$. Or: $C_{ij} = \text{row}_i(A) \cdot \text{col}_j(B)$.

**Why define it this way?** Because it represents **composition of transformations**. First transform with $B$, then with $A$. Order matters: $AB \neq BA$ (non-commutative).

### 3.4 Transpose and Inverse

- **Transpose** $A^T$: swap rows and columns. $(A^T)_{ij} = A_{ji}$.
- **Inverse** $A^{-1}$: $A \cdot A^{-1} = I$ (identity). Exists only if $A$ is square and **full-rank**.

**What does the inverse do?** It undoes the transformation. If $w = A v$, then $v = A^{-1} w$. In linear regression: $\hat{w} = (X^T X)^{-1} X^T y$ — we are "undoing" the mixing to find the coefficients.

### 3.5 Rank

The **rank** of a matrix is the number of linearly independent rows/columns. It measures how much information the matrix truly contains.

- **Rank-deficient**: if you have 1000 features but many are linear combinations of others, the effective rank is much lower.
- **Full rank**: every row/column contributes new information.
- A matrix of rank $r$ can be approximated by matrices of lower rank (SVD gives the best approximation).

### 3.6 Trace

$$\text{tr}(A) = \sum_i A_{ii}$$

Sum of the diagonal. Appears in properties like $\text{tr}(ABC) = \text{tr}(BCA)$ (cyclic) and in computing total variance explained in PCA.

---

## 4. Decompositions — The Heart of Linear Algebra for ML

### 4.1 Eigenvalues and Eigenvectors

$$A v = \lambda v$$

$v$ is an **eigenvector** of $A$, and $\lambda$ is its **eigenvalue**.

**Intuition**: when you apply the transformation $A$ to $v$, the result points in the **same direction**. It only stretches (or shrinks) by factor $\lambda$.

Imagine a matrix that stretches 2D space: the X-axis doubles, the Y-axis stays. The eigenvectors are $[1,0]$ (with $\lambda=2$) and $[0,1]$ (with $\lambda=1$).

**Direct application — PCA**:
1. Compute covariance matrix $\Sigma = \frac{1}{n} X^T X$
2. Its eigenvectors are the **directions of maximum variance**
3. Its eigenvalues say **how much variance each direction explains**
4. Take the $k$ eigenvectors with largest $\lambda$ and project the data → dimensionality reduction

**Spectral Theorem**: if $A$ is symmetric ($A = A^T$, like covariance), its eigenvectors are orthogonal. This makes everything more manageable.

### 4.2 Singular Value Decomposition (SVD)

$$A = U \Sigma V^T$$

This is the **most powerful** of all decompositions.

**Geometric intuition**: every linear transformation decomposes into three steps:
1. **Rotation** ($V^T$) — reorient space
2. **Scaling** ($\Sigma$) — stretch/shrink each axis by its singular value
3. **Another rotation** ($U$) — rotate to the final coordinate system

**Concrete example**:
```
A = [3 1]
    [1 3]

SVD:
U = [0.707 -0.707]    Σ = [4 0]    V^T = [0.707  0.707]
    [0.707  0.707]        [0 2]          [-0.707 0.707]
```

The singular values are $\sigma_1 = 4$, $\sigma_2 = 2$. The first explains $\frac{4}{4+2} = 66.7\%$ of the "energy" of the matrix.

**Why is SVD so important?**

| Application | How it uses SVD |
|---|---|
| **PCA** | PCA = SVD of the centered matrix. Components = $V$, variance = $\Sigma^2$ |
| **Recommendation** | Netflix Prize: factorize user-item matrix with truncated SVD |
| **Compression** | Keep only the $k$ largest singular values (10× smaller, slightly lower quality) |
| **Denoising** | Noise lives in small components; removing them = filtering |
| **LLM Embeddings** | Word2vec, GloVe — factorizing co-occurrence matrices is SVD |
| **Matrix Completion** | Predict missing entries (as in recommendation) |

### 4.3 Cholesky Decomposition

$$A = L L^T$$

For symmetric positive-definite matrices (like covariance).

**Why is it useful?**
- Solve linear systems fast ($Ax = b$ in $O(n^2)$ with Cholesky vs $O(n^3)$ with inverse)
- Sample from multivariate Gaussian distributions
- Gaussian Processes

---

## 5. Common Mistakes

1. **Not standardizing before PCA**: PCA maximizes variance. If "age" (0-100) and "income" (0-1M) are on different scales, the first component will essentially be "income". Always standardize (mean 0, variance 1).

2. **Confusing eigenvectors of $X$ with eigenvectors of $X^T X$**: PCA operates on the covariance $X^T X$, not on $X$ directly. The eigenvectors of $X^T X$ are the principal directions.

3. **SVD is not PCA**: PCA is SVD applied to the centered matrix. If you do not center the data, SVD gives directions of maximum "energy" but not necessarily maximum variance.

4. **Non-invertible matrix ≠ useless matrix**: It means there is linear dependence. Regularization (Ridge) solves this.

5. **Forgetting that $QK^T$ in attention is a dot product matrix**: Each entry $(i,j)$ is $q_i \cdot k_j$. When $q_i$ and $k_j$ are orthogonal, attention is zero for that pair.

---

## 6. Check Your Understanding

1. Given two vectors $v = [1, 0]$ and $w = [0, 1]$, what is their dot product? What does this tell you about the angle between them?
2. Why is the covariance matrix $X^T X$ symmetric? And why are its eigenvectors orthogonal?
3. If you SVD a matrix and all singular values are equal, what shape is the matrix?
4. In a 2-layer neural network: $h = W_1 x$, $y = W_2 h$, what does each matrix $W$ represent geometrically?
5. You have 100 samples and 1000 features. The effective rank is 80. What does this mean and how does it affect your model?

---

## 7. Summary

Linear algebra is the language of machine learning. Your data is a collection of vectors. Transformations (matrices) are your models. Decompositions (SVD, eigenvalues) are the tools that let you see what is really happening. When you see a transformer, remember: $QK^T$ is a matrix of dot products measuring token similarity. When you run PCA, remember: you are finding the eigenvectors of the covariance. Everything is linear algebra.

---

## 8. Where to Go Next

- [[Calculus]] — Optimization requires derivatives
- [[Probability]] — Distributions live in vector spaces
- [[Statistics]] — Covariance is a linear algebra concept
- [[Neural Networks]] — Every layer is $Wx + b$
- [[Transformers]] — Self-attention = dot products between queries and keys
- [[Unsupervised Learning]] — PCA, t-SNE, K-Means all use linear algebra heavily
- [[Gradient-Based Optimization]] — Gradient descent updates are matrix operations
