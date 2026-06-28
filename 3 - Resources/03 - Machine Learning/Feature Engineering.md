---
tags: [machine-learning, features, core]
status: growing
created: 2026-06-27
---

# Feature Engineering

## 1. Why This Matters

"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." — Andrew Ng

The best model with bad features will lose to a mediocre model with great features. Feature engineering is where domain knowledge meets data science. It is the process of transforming raw data into inputs that make ML algorithms work effectively.

---

## 2. The Feature Engineering Pipeline

```
Raw Data → Cleaning → Transformation → Encoding → Creation → Selection → Model
```

---

## 3. Handling Missing Values

### 3.1 Why Missing Values Matter

Most ML algorithms cannot handle NaN values natively. You must decide what to do.

### 3.2 Strategies

| Strategy | Method | When to Use |
|---|---|---|
| **Drop rows** | `df.dropna()` | Few missing (<5%), MCAR (missing completely at random) |
| **Drop columns** | `df.drop(columns=[...])` | High missing rate (>70%), low predictive value |
| **Mean/Median impute** | `df.fillna(df.mean())` | Numeric data, missing at random |
| **Mode impute** | `df.fillna(df.mode())` | Categorical data |
| **Forward fill** | `df.fillna(method="ffill")` | Time series |
| **Model-based** | KNN imputer, IterativeImputer | Complex patterns, enough data |

### 3.3 Missingness as Information

Sometimes *why* a value is missing matters. Create a binary feature: `is_age_missing`.

---

## 4. Encoding Categorical Variables

### 4.1 Ordinal Encoding

For categories with a natural order:

```python
from sklearn.preprocessing import OrdinalEncoder

education_order = ["high_school", "bachelors", "masters", "phd"]
encoder = OrdinalEncoder(categories=[education_order])
X_encoded = encoder.fit_transform(X[["education"]])
```

### 4.2 One-Hot Encoding

For categories without order. Creates $k$ binary columns for $k$ categories.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[["color", "city"]])
```

**Warning**: if a column has 1000 categories, one-hot encoding creates 1000 new columns. This is called the **curse of dimensionality**. Alternatives:
- **Target encoding**: replace category with mean target for that category
- **Frequency encoding**: replace category with its count/frequency
- **Feature hashing**: map categories to a fixed number of dimensions

### 4.3 Target Encoding

Replace each category with the mean of the target for rows in that category:

```python
# Simple (but prone to overfitting — use smoothing)
mean_encoded = df.groupby("category")["target"].mean()

# With smoothing (borrows from global mean for rare categories)
global_mean = df["target"].mean()
smoothing = 10  # higher = more shrinkage toward global mean
counts = df.groupby("category")["target"].count()
means = df.groupby("category")["target"].mean()
smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
```

---

## 5. Numerical Transformations

### 5.1 Scaling

| Scaler | Formula | Robust to outliers? | Output range |
|---|---|---|---|
| **StandardScaler** | $z = (x - \mu) / \sigma$ | No | Theoretical (-∞, ∞) |
| **MinMaxScaler** | $x' = (x - min) / (max - min)$ | No | [0, 1] |
| **RobustScaler** | $x' = (x - median) / IQR$ | Yes | Theoretical (-∞, ∞) |
| **MaxAbsScaler** | $x' = x / | max|$ | No | [-1, 1] |

**When each matters**:
- **Tree-based models** (RF, XGBoost): scaling is NOT needed (splits are threshold-based)
- **Distance-based models** (KNN, SVM, linear models): scaling is ESSENTIAL
- **Neural networks**: scaling is strongly recommended for stable training

### 5.2 Handling Skew

Many ML models perform better when features are roughly normally distributed:

```python
import numpy as np

# Log transform (for positive data with right skew)
X["income_log"] = np.log1p(X["income"])  # log(1 + x)

# Box-Cox (automatically finds best transformation)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="box-cox")
X_transformed = pt.fit_transform(X)

# Yeo-Johnson (like Box-Cox but works with zero/negative values)
pt = PowerTransformer(method="yeo-johnson")
```

### 5.3 Binning (Discretization)

Convert continuous features into categorical buckets:

```python
X["age_group"] = pd.cut(X["age"], bins=[0, 18, 30, 50, 100],
                        labels=["child", "young", "middle", "senior"])
```

Useful when the relationship between the feature and target is non-monotonic (e.g., age vs risk of certain diseases).

---

## 6. Feature Creation

### 6.1 Domain-Specific Features

The most powerful features often come from domain knowledge:

```python
# Ratio features
df["debt_to_income"] = df["total_debt"] / df["income"]

# Difference features
df["days_since_last_purchase"] = (reference_date - df["last_purchase_date"]).dt.days

# Aggregation features
df["avg_purchase_by_customer"] = df.groupby("customer_id")["purchase_amount"].transform("mean")

# Date features
df["is_weekend"] = df["date"].dt.dayofweek >= 5
df["hour"] = df["timestamp"].dt.hour
df["day_of_year"] = df["date"].dt.dayofyear
```

### 6.2 Interaction Features

Capture non-linear relationships between features:

```python
from sklearn.preprocessing import PolynomialFeatures

# Creates x1, x2, x1², x1*x2, x2²
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# Or manually for specific interactions
df["age_income_interaction"] = df["age"] * df["income"]
```

Be careful — polynomial features explode exponentially with degree and number of features.

### 6.3 Text Features

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bag of words (word counts)
vectorizer = CountVectorizer(max_features=1000, stop_words="english")

# TF-IDF (word frequency weighted by inverse document frequency)
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
```

---

## 7. Feature Selection

### 7.1 Filter Methods

Rank features independently of the model:

```python
# Variance threshold (remove constant/near-constant features)
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)

# Correlation with target (see [[Statistics]])
correlations = df.corr()["target"].abs().sort_values(ascending=False)

# Mutual information
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X, y)
```

### 7.2 Wrapper Methods

Train models with different feature subsets:

```python
from sklearn.feature_selection import RFE

# Recursive Feature Elimination
selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=20)
selector.fit(X, y)
selected_features = X.columns[selector.support_]
```

### 7.3 Embedded Methods

Feature selection happens during model training (see [[Regularization]] for L1/L2 details):

```python
# Lasso (L1) — coefficients become exactly zero
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]

# Tree-based importance
importances = rf.feature_importances_
top_k_idx = importances.argsort()[-20:][::-1]
```

---

## 8. The Feature Pipeline

Always build a modular pipeline to avoid data leakage (see [[Python for Data Science]] for sklearn best practices):

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier()),
])

model.fit(X_train, y_train)  # fit_transform on train, transform on test
```

---

## 9. Common Mistakes

1. **Data leakage in feature engineering**: computing `df.mean()` on the whole dataset before splitting — the mean "knows" the test data. Always compute statistics on training data only.

2. **Creating too many features**: 100 samples with 1000 features will almost certainly overfit. Feature selection is not optional in high dimensions.

3. **Assuming more features = better**: irrelevant features add noise, not signal. A model with 10 good features beats a model with 1000 mediocre features.

4. **Not handling rare categories in one-hot encoding**: a category that appears once in training and never in test (or vice versa) causes errors. Use `handle_unknown="ignore"`.

5. **Applying log transform to negative or zero values**: `log(0)` is undefined. Use `np.log1p(x)` or Yeo-Johnson transform.

---

## 10. Check Your Understanding

1. Why does a tree-based model not need scaled features, while SVM does?
2. You have a categorical column with 5000 unique values. What encoding strategy do you use?
3. How can forward-fill (`method="ffill"`) cause data leakage in time series?
4. A feature has correlation 0.8 with the target. Is it necessarily a good feature? What could go wrong?
5. You engineer 500 features from 2000 samples. What problem are you likely to encounter?

---

## 11. Summary

Feature engineering is where domain knowledge meets data science. Good features make simple models perform well. The process involves handling missing data, encoding categories, scaling numerics, creating interaction features, and selecting the most informative subset. Always wrap feature engineering in a pipeline to prevent data leakage. The golden rule: compute statistics on training data only, then transform test data using those statistics.

---

## 12. Where to Go Next

- [[Supervised Learning]] — Using engineered features in models
- [[Model Evaluation]] — How feature selection affects bias-variance
- [[Unsupervised Learning]] — PCA as automated feature extraction
