---
tags: [machine-learning, features, core]
status: growing
created: 2026-06-27
---

# Feature Engineering

## 1. Escenario de aprendizaje

Trabajas en una startup de fintech y tienes datos crudos de transacciones: montos, fechas, ubicaciones y tipos de comercio. Tu tarea es predecir si una transacción es fraudulenta. Pero los algoritmos de machine learning no entienden "fecha" o "ubicación" directamente — necesitas transformar estos datos en características numéricas que capturen patrones de fraude. El mejor modelo con malas características perderá contra un modelo mediocre con características excelentes. Feature engineering es donde el conocimiento del dominio se encuentra con la ciencia de datos.

---

## 2. The Feature Engineering Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Data   │───▶│  Cleaning   │───▶│Transform.   │───▶│  Encoding   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                    │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   Model     │◀───│ Selection   │◀───│  Creation   │◀────────────┘
└─────────────┘    └─────────────┘    └─────────────┘
```

### Pasos del Pipeline:

| Paso | Descripción | Ejemplo |
|------|-------------|---------|
| **1. Limpieza** | Manejar valores faltantes, duplicados | `dropna()`, `fillna()` |
| **2. Transformación** | Escalar, normalizar, manejar sesgo | `StandardScaler`, `log1p()` |
| **3. Encoding** | Convertir categorías a numéricos | One-Hot, Target Encoding |
| **4. Creación** | Generar nuevas features | Interacciones, ratios |
| **5. Selección** | Elegir las mejores features | `SelectKBest`, `RFE` |

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

### Comparación de Métodos de Encoding

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MÉTODOS DE ENCODING                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ORDINAL ENCODING          ONE-HOT ENCODING         TARGET ENCODING │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐│
│  │ low  → 0     │         │ red  → 1,0,0  │         │ red  → 0.85  ││
│  │ med  → 1     │         │ blue → 0,1,0  │         │ blue → 0.42  ││
│  │ high → 2     │         │ green→ 0,0,1  │         │ green→ 0.67  ││
│  └──────────────┘         └──────────────┘         └──────────────┘│
│                                                                     │
│  ✓ Categorías con orden   ✓ Sin orden             ✓ Alta cardinalidad│
│  ✓ Solo 1 columna         ✗ Muchas columnas       ✓ 1 columna      │
│  ✗ Asume distancia igual  ✗ Muy disperso          ✗ Puede causar   │
│                                                  leakage           │
└─────────────────────────────────────────────────────────────────────┘
```

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

#### Comparación Visual de Métodos de Escalado

```
ANTES DE ESCALAR:                    DESPUÉS DE ESCALAR:
                                      
   │    •                              │      •    •
   │  •   •                            │  •        •
   │    • •                            │    •    •
   │  •   •                            │  •    •
   │• •                                │•     •
   └──────────────                     └──────────────
   [0, 100000]                         [0, 1]
   
StandardScaler        MinMaxScaler        RobustScaler
z = (x - μ) / σ       x' = (x-min)/(max-min)   x' = (x-median)/IQR
Media=0, Var=1        Rango [0,1]         Robusto a outliers
```

| Scaler | Formula | ¿Robusto a outliers? | Rango de salida |
|---|---|---|---|
| **StandardScaler** | $z = (x - \mu) / \sigma$ | No | Teórico (-∞, ∞) |
| **MinMaxScaler** | $x' = (x - min) / (max - min)$ | No | [0, 1] |
| **RobustScaler** | $x' = (x - median) / IQR$ | Sí | Teórico (-∞, ∞) |
| **MaxAbsScaler** | $x' = x / | max|$ | No | [-1, 1] |

**¿Cuándo usar cada método?**:
- **Modelos basados en árboles** (RF, XGBoost): NO necesitan escalado (los splits son por umbral)
- **Modelos basados en distancia** (KNN, SVM, modelos lineales): escalado es ESENCIAL
- **Redes neuronales**: escalado es muy recomendado para entrenamiento estable

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

### Métodos de Selección de Features

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MÉTODOS DE SELECCIÓN                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FILTER METHODS         WRAPPER METHODS         EMBEDDED METHODS    │
│  ┌──────────────┐      ┌──────────────┐        ┌──────────────┐   │
│  │ Correlación  │      │ RFE          │        │ Lasso (L1)   │   │
│  │ Mutual Info  │      │ Forward/Back │        │ Ridge (L2)   │   │
│  │ Chi-cuadrado │      │ Stepwise     │        │ Árboles      │   │
│  └──────────────┘      └──────────────┘        └──────────────┘   │
│                                                                     │
│  ✓ Rápido               ✓ Considera interacciones  ✓ Integrado     │
│  ✓ Modelo-agnóstico     ✗ Muy lento               ✗ Modelo-       │
│  ✗ Ignora interacciones ✗ Costoso computacionalmente  específico   │
└─────────────────────────────────────────────────────────────────────┘
```

### Flujo de Decisión

```
¿Muchas features?
    │
    ▼ Sí
¿Cuántas? ──── <100 ───▶ Usar Filter Methods (correlación, mutual info)
    │
    ▼ >100
¿Modelo lineal? ── Sí ──▶ Usar Lasso (L1) para selección automática
    │
    ▼ No
¿Necesitas interpretabilidad? ── Sí ──▶ Usar RFE con Random Forest
    │
    ▼ No
Usar Embedded Methods (feature importance de árboles)

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

## 9. Errores Comunes

1. **Data leakage en feature engineering**: calcular `df.mean()` en todo el dataset antes de dividir — la media "conoce" los datos de prueba. Siempre calcula estadísticas solo en los datos de entrenamiento.

2. **Crear demasiadas features**: 100 muestras con 1000 features casi siempre causarán overfitting. La selección de features no es opcional en dimensiones altas.

3. **Asumir que más features = mejor**: features irrelevantes añaden ruido, no señal. Un modelo con 10 buenas features supera a uno con 1000 features mediocres.

4. **No manejar categorías raras en one-hot encoding**: una categoría que aparece una vez en entrenamiento y nunca en prueba (o viceversa) causa errores. Usa `handle_unknown="ignore"`.

5. **Aplicar transformación logarítmica a valores negativos o cero**: `log(0)` no está definido. Usa `np.log1p(x)` o la transformación Yeo-Johnson.

---

## 10. Comprueba tu Conocimiento

1. ¿Por qué un modelo basado en árboles no necesita features escaladas, mientras que SVM sí?
2. Tienes una columna categórica con 5000 valores únicos. ¿Qué estrategia de encoding usas?
3. ¿Cómo puede el llenado hacia adelante (`method="ffill"`) causar data leakage en series temporales?
4. Una feature tiene correlación 0.8 con el target. ¿Es necesariamente buena? ¿Qué podría salir mal?
5. Creas 500 features a partir de 2000 muestras. ¿Qué problema probablemente encontrarás?

---

## 11. Resumen

Feature engineering es donde el conocimiento del dominio se encuentra con la ciencia de datos. Buenas features hacen que modelos simples rindan bien. El proceso implica manejar datos faltantes, codificar categorías, escalar numéricos, crear features de interacción y seleccionar el subconjunto más informativo. Siempre envuelve feature engineering en un pipeline para evitar data leakage. La regla de oro: calcula estadísticas solo en datos de entrenamiento, luego transforma los datos de prueba usando esas estadísticas.

---

## 12. ¿Dónde ir Siguente?

- [[Supervised Learning]] — Usar features ingenierizadas en modelos
- [[Model Evaluation]] — Cómo la selección de features afecta bias-variance
- [[Unsupervised Learning]] — PCA como extracción automatizada de features
- [[Regularization]] — Cómo L1/L2 seleccionan features automáticamente
