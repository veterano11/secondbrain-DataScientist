---
tags: [machine-learning, supervised, core]
status: growing
created: 2026-06-27
---

# Supervised Learning

## 1. Escenario de aprendizaje

Imagina que trabajas en un banco y te piden predecir si un cliente va a dejar el servicio (churn) con base en su historial de transacciones, llamadas al soporte y productos contratados. Tienes miles de ejemplos de clientes pasados donde ya sabes si se fueron o se quedaron. Eso es supervised learning: aprendes un mapeo de entradas $x$ (datos del cliente) a salidas $y$ (churn o no churn) usando ejemplos etiquetados $(x_i, y_i)$. Pero la simplicidad del concepto esconde la profundidad de los algoritmos, la sutileza de la evaluación y el cuidado necesario para evitar overfitting.

---

## 2. The Supervised Learning Framework

### 2.1 Formal Definition

Given:
- Training data $(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)$
- $x_i \in \mathcal{X}$ (feature space)
- $y_i \in \mathcal{Y}$ (label space: $\mathbb{R}$ for regression, $\{1,...,K\}$ for classification)

Find $f: \mathcal{X} \to \mathcal{Y}$ that minimizes expected loss on new data.

### 2.2 The Two Main Branches

| Branch | $y$ type | Example |
|---|---|---|
| **Regression** | Continuous | Predict house price, temperature, sales |
| **Classification** | Discrete | Detect spam, classify images, diagnose disease |

---

## 3. Linear Models

### 3.1 Linear Regression

The simplest and most interpretable model. Assumes a linear relationship between features and target:

$$y = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b + \epsilon = Xw + \epsilon$$

**Geometric intuition**: find a hyperplane that best fits the data points.

**The Normal Equation** (closed-form solution — see [[Linear Algebra]] for matrix inversion details):

$$\hat{w} = (X^T X)^{-1} X^T y$$

**Step-by-step**:
1. $X^T X$: compute the covariance-like matrix (d × d)
2. $(X^T X)^{-1}$: invert it (requires full rank — no multicollinearity)
3. $X^T y$: compute correlation between features and target
4. Multiply: $\hat{w}$ gives the optimal weights

**Concrete example** — predicting salary from years experience:
```
Data: (1yr, 40k), (2yr, 45k), (3yr, 50k), (4yr, 55k)

X = [[1], [2], [3], [4]]
y = [40000, 45000, 50000, 55000]

w = (X^T X)^(-1) X^T y
  = (sum x_i²)^(-1) × sum(x_i y_i)
  = 1/30 × 150000 = 5000

So: salary ≈ 35000 + 5000 × years_experience
```

**Assumptions** (worth knowing because violating them degrades performance — see [[Statistics]] for deeper context):
1. **Linearity**: relationship between features and target is linear
2. **Independence**: observations are independent (no autocorrelation)
3. **Homoscedasticity**: constant variance of errors across all x values
4. **Normality**: errors are normally distributed (for inference, not prediction)

When these assumptions are violated, use regularization (Ridge/Lasso), non-linear models, or transformations.

### 3.2 Logistic Regression

Despite the name, it is a **classification** algorithm. It models the probability of belonging to a class:

$$P(y=1|x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$

**Why sigmoid?** Linear regression outputs unbounded values $(-\infty, \infty)$. A probability must be in $[0, 1]$. The sigmoid function squashes any real number into this range:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

- When $z \to \infty$: $\sigma(z) \to 1$
- When $z \to -\infty$: $\sigma(z) \to 0$
- When $z = 0$: $\sigma(0) = 0.5$ (decision boundary)

**Decision boundary**: the set of points where $P(y=1|x) = 0.5$, i.e., $w^T x = 0$. This is a linear surface.

**Multiclass extension**: softmax regression (also called multinomial logistic regression):

$$P(y=k|x) = \frac{e^{w_k^T x}}{\sum_{j=1}^K e^{w_j^T x}}$$

---

## 4. Tree-Based Models

### 4.1 Decision Trees

A decision tree splits data recursively based on feature values.

**How it works** (step by step):
1. Look at all features and all possible split points
2. Choose the split that best separates the target (lowest impurity)
3. Repeat recursively on each partition
4. Stop when max depth is reached, min samples per leaf is met, or no split improves purity

**Impurity metrics**:
- **Gini**: $2p(1-p)$ for binary classification
- **Entropy**: $-p\log p - (1-p)\log(1-p)$
- **MSE** (regression): variance of target in the node

**Intuition**: imagine sorting emails by "contains word 'free'" — the first split separates it into two groups, one with mostly spam (left) and one with mostly ham (right). Now split the left group by "contains word 'urgent'", and so on.

### 4.2 Random Forest

Builds many decision trees and averages their predictions.

**Why it works**: each tree is trained on a different bootstrap sample of the data (bagging), and each split considers only a random subset of features. This decorrelates the trees. The average of many imperfect trees is more stable and accurate than any single tree.

- **Reduces variance** without increasing bias significantly
- **Handles non-linearity** naturally
- **Feature importance**: features used near the top of many trees are more important

### 4.3 Gradient Boosting (XGBoost, LightGBM)

**Intuition**: instead of averaging many independent trees (Random Forest), build trees **sequentially**, where each tree tries to correct the errors of the previous ones.

1. Start with a simple prediction (e.g., mean of target)
2. Compute residuals (errors) of current prediction
3. Train a small tree to predict the residuals
4. Add the tree's prediction to the ensemble (with a learning rate)
5. Repeat steps 2-4 hundreds or thousands of times

XGBoost adds regularization to the trees, handles missing values, and is heavily optimized for performance via [[Gradient-Based Optimization]]. It is the go-to algorithm for tabular data competitions.

---

## 5. Support Vector Machines (SVM)

**Core idea**: find the hyperplane that separates classes with the **maximum margin**.

The margin is the distance from the hyperplane to the nearest points of each class (the support vectors). Maximizing the margin improves generalization.

**Kernel trick**: map data to a higher-dimensional space where it becomes linearly separable, without explicitly computing the mapping:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

Common kernels:
- **Linear**: $K(x_i, x_j) = x_i^T x_j$ — no mapping, just linear SVM
- **RBF** (Gaussian): $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ — infinite-dimensional mapping
- **Polynomial**: $K(x_i, x_j) = (x_i^T x_j + c)^d$

SVMs work well when the number of features is large relative to samples (e.g., text classification with bag-of-words).

---

## 6. K-Nearest Neighbors (KNN)

The simplest algorithm: store all training data. To predict a new point, find the $k$ closest training points and vote.

- **No training** (lazy learner) — just memorize the data
- Prediction is O(n) — must compute distance to every training point
- Sensitive to feature scaling (use StandardScaler)
- Works best with few features (curse of dimensionality)

---

## 7. Evaluating Supervised Models

Train/test split is non-negotiable:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # for classification
)
```

For reliable evaluation:
- Always use a held-out test set (not just validation)
- Use cross-validation for [[Hyperparameter Tuning]]
- Stratify classification splits to preserve class proportions

---

## 8. Common Mistakes

1. **Data leakage**: using information from the test set during training. Common forms: scaling before splitting, using future data to predict the past, group-level features without group splitting.

2. **Training on imbalanced data without care**: 99% accuracy on a 99:1 imbalance is meaningless. Use class weights, resampling (SMOTE), or different metrics (F1, precision-recall AUC).

3. **Using accuracy for imbalanced problems**: a model that predicts "no disease" for everyone achieves 99% accuracy when disease prevalence is 1%. Always check the confusion matrix.

4. **Not checking model assumptions**: linear regression assumes linearity and homoscedasticity. If violated, predictions can be systematically biased.

5. **Overfitting before seeing test data**: tuning hyperparameters on the test set invalidates it. Use a separate validation set or cross-validation.

---

## 9. Check Your Understanding

1. Why does Lasso regression (L1) produce sparse coefficients (many exactly zero) while Ridge (L2) does not?
2. A decision tree of depth 10 has at most how many leaf nodes? How many parameters?
3. Why does Random Forest reduce variance compared to a single decision tree?
4. Logistic regression outputs a probability. You need a binary decision. Where do you set the threshold? What tradeoffs does the threshold control?
5. You have 10 features and 50 samples. Which algorithms are most/least suitable and why?

---

## 10. Summary

Supervised learning learns a mapping from inputs to outputs using labeled data. Linear models offer simplicity and interpretability (regression, logistic). Tree-based models handle non-linearity and interactions naturally (Random Forest, XGBoost). The key challenge is generalization — the model must perform well on data it has never seen. Proper evaluation (train/test split, cross-validation) and awareness of bias-variance tradeoffs separate effective practitioners from those who overfit to noise.

---

## 11. Where to Go Next

- [[Unsupervised Learning]] — Finding structure without labels
- [[Model Evaluation]] — Metrics, validation strategies, and best practices
- [[Feature Engineering]] — Creating features that make models work
- [[Regularization]] — Preventing overfitting in depth
