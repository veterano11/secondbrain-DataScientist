---
tags: [programming, python, data-science, applied]
status: growing
created: 2026-06-27
---

# Python for Data Science

## 1. Why This Matters

Python fundamentals are necessary but not sufficient. The data science ecosystem — NumPy, pandas, scikit-learn, PyTorch — is what makes Python the dominant language for ML. Understanding these libraries deeply (not just how to use them, but **how they work**) lets you write faster, more correct, and more maintainable data code.

---

## 2. NumPy — The Foundation

NumPy provides the **ndarray** (n-dimensional array), which is faster and more memory-efficient than Python lists for numerical data.

### 2.1 Why NumPy is Fast

Python lists contain pointers to objects — each element is a Python object. NumPy arrays store **raw C values** in contiguous memory.

```python
# Python list: each element is a PyObject pointer
py_list = [1, 2, 3]          # ~72 bytes per element

# NumPy array: raw int64 values in a contiguous block
np_array = np.array([1, 2, 3])  # 8 bytes per element + overhead
```

This means:
- **Less memory**: 8 bytes per int vs ~72 bytes
- **CPU cache friendly**: sequential memory access is faster
- **Vectorized operations**: operations run in C, not Python loops

### 2.2 Creating Arrays

```python
import numpy as np

arr = np.array([1, 2, 3])
zeros = np.zeros((3, 4))           # 3 rows, 4 cols
ones = np.ones((2, 3))
eye = np.eye(3)                    # identity matrix (3×3)
linspace = np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1.0]
random = np.random.randn(1000)     # 1000 samples from N(0, 1)
```

### 2.3 Vectorization — The Key Concept

**Without vectorization** (slow):
```python
result = np.zeros(1000)
for i in range(1000):
    result[i] = a[i] * b[i] + c[i]  # Python loop — interpreted, slow
```

**With vectorization** (fast):
```python
result = a * b + c                   # No Python loop — runs in C
```

**Rule**: never loop over NumPy arrays in Python. Use vectorized operations. [[Data Structures & Algorithms]] explains the Big O concepts behind this performance advantage.

### 2.4 Broadcasting

Operations between arrays of different shapes are automatically aligned:

```python
matrix = np.ones((3, 4))      # 3×4
row_mean = matrix.mean(axis=1)  # shape (3,) — mean of each row
centered = matrix - row_mean[:, np.newaxis]  # broadcasts across columns

# Broadcasting rules:
# (3, 4) and (3,) → insert dimension → (3, 4) and (3, 1) → (3, 4)
```

### 2.5 Linear Algebra

```python
# Matrix multiplication
C = A @ B                      # same as np.matmul(A, B)
v = W @ x + b                  # neural network layer

# Decompositions
eigvals, eigvecs = np.linalg.eig(cov_matrix)
U, S, Vt = np.linalg.svd(matrix)
```

---

## 3. pandas — Data Manipulation

pandas is built on top of NumPy and adds labeled rows and columns.

### 3.1 The DataFrame

A DataFrame is conceptually a **dictionary of Series** (columns). Each column has a single type.

```python
import pandas as pd

df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json")

# Basic inspection
df.info()          # columns, types, non-null counts
df.describe()      # summary statistics
df.head()          # first 5 rows
df.shape           # (rows, columns)
```

### 3.2 Selection — .loc vs .iloc

This is the most common source of confusion.

```python
# .loc: label-based (column names, index labels)
df.loc[rows, columns]
df.loc[df["age"] > 30, ["name", "income"]]  # boolean row mask

# .iloc: integer position-based
df.iloc[row_indices, column_indices]
df.iloc[10:20, [0, 3, 5]]  # rows 10-19, columns 0, 3, 5
```

**When to use each**: `.loc` when you know column names (most of the time), `.iloc` when you are iterating programmatically.

### 3.3 The Split-Apply-Combine Pattern

This is the most important data pattern in pandas — it mirrors the [[Functional Programming]] map-filter-reduce paradigm:

```python
# Group by category → apply function → combine results
df.groupby("department")["salary"].agg(["mean", "std", "count"])

# Step by step:
# 1. Split: partition data by department
# 2. Apply: compute mean, std, count for each
# 3. Combine: merge results into a new DataFrame

# More complex
result = (df
    .groupby(["department", "year"])
    .agg(
        avg_salary=("salary", "mean"),
        headcount=("employee_id", "nunique"),
        total_bonus=("bonus", "sum"),
    )
    .reset_index())
```

### 3.4 Merging and Joining

```python
# SQL-style joins
pd.merge(orders, customers, on="customer_id", how="inner")
pd.merge(orders, customers, on="customer_id", how="left")

# Concatenation
pd.concat([df1, df2], axis=0)  # stack rows (len grows)
pd.concat([df1, df2], axis=1)  # side by side (columns grow)
```

### 3.5 Performance Tips

- **Avoid `apply` with Python functions** for large datasets (it loops in Python). Use vectorized operations instead.
- **Use categorical dtype** for low-cardinality string columns (saves memory, faster groupbys).
- **Use inplace=False** (the default) — it is safer and often equally fast.
- ** `query()` is more readable** for complex filters: `df.query("age > 30 and income > 50000")`

---

## 4. scikit-learn — The ML Interface

scikit-learn established the consistent `fit` / `predict` / `transform` API that became standard across ML in Python.

### 4.1 The API Pattern

Every model and transformer follows the same pattern, a classic application of [[Object-Oriented Programming]] polymorphism:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100)),
])

pipeline.fit(X_train, y_train)        # learn parameters
predictions = pipeline.predict(X_test)  # apply learned model
```

### 4.2 Why the Uniform API Matters

- **Interchangeable components**: swap `RandomForestClassifier` for `XGBClassifier` without changing anything else
- **Grid search**: `GridSearchCV` works on any model with the same API
- **Pipelines**: compose preprocessing + modeling into one object
- **Production**: exporters and serving frameworks expect the API

### 4.3 Essential Tools

```python
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV)
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   LabelEncoder)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
```

---

## 5. PyTorch vs TensorFlow — A Quick Orientation

Both are deep learning frameworks. PyTorch has become the dominant choice for research and increasingly for production.

**PyTorch key philosophy**: define-by-run (dynamic computation graphs).

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

---

## 6. Common Mistakes

1. **Chained indexing**: `df[df["age"] > 30]["income"]` — this may return a copy or a view. Use `df.loc[df["age"] > 30, "income"]` instead.

2. **Forgetting that `apply` is a Python loop**: `df["col"].apply(complex_python_function)` will be slow. Vectorize if possible.

3. **Modifying a DataFrame you are iterating over**: "SettingWithCopyWarning" is a symptom. Use `.loc` or a copy.

4. **Not using categorical data**: strings repeated thousands of times waste memory. `df["col"] = df["col"].astype("category")`.

5. **Writing explicit loops over NumPy arrays**: almost always, there is a vectorized alternative that is 50-100× faster.

---

## 7. Check Your Understanding

1. Why is `np.sum(a * b)` faster than `sum(x * y for x, y in zip(a, b))`?
2. What is the difference between `df.loc[5]` and `df.iloc[5]`?
3. You have a DataFrame with 50 million rows and a column of country codes (200 unique values). How can you reduce memory usage?
4. Why does scikit-learn use the `fit()`/`transform()` API instead of functions?
5. What does `df.groupby("A")["B"].mean()` do? Write it step by step.

---

## 8. Summary

NumPy provides fast numerical arrays with vectorized operations. pandas adds labeled data on top. scikit-learn provides a consistent ML API. The key insight: **avoid Python loops**, use vectorized operations, and follow the uniform `fit`/`predict` API. These three libraries and their patterns form the backbone of data science in Python.

---

## 9. Where to Go Next

- [[Python Fundamentals]] — Core Python needed to understand libraries
- [[Feature Engineering]] — pandas and sklearn for feature creation
- [[Neural Networks]] — PyTorch for deep learning
