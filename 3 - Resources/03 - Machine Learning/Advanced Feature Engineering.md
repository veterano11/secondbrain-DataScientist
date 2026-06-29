---
tags:
  - feature-engineering
  - machine-learning
  - advanced
status: seedling
created: 2026-06-28
---

# Advanced Feature Engineering

## 1. Escenario de aprendizaje

Gastaste 3 semanas construyendo features para un modelo de predicción de churn. Ingeniería manual: ratios, agregados por usuario, ventanas de tiempo. Cuando pruebas el modelo, el lift contra el baseline sin features es apenas 2%. No porque las features sean malas — sino porque aplicaste las técnicas equivocadas a los tipos de datos equivocados.

Tu jefe te dice: "El modelo de [[Supervised Learning]] que teníamos antes ya funcionaba bien. ¿Qué ganamos realmente con esto?" No sabes qué responder porque no mediste el impacto de cada técnica por separado.

Esta nota cubre técnicas avanzadas de [[Feature Engineering]] ordenadas por tipo de dato y problema, con énfasis en validación rigurosa y detección de data leakage.

---

## 2. Target Encoding

Target encoding reemplaza cada categoría de una variable categórica por la media de la variable objetivo para esa categoría. Es poderoso para cardinalidad alta (cientos o miles de categorías) pero extremadamente propenso a data leakage.

### 2.1 Fórmula básica

Para una categoría `k` con `n_k` observaciones y `y_k` suma del target:

```
encoded(k) = mean(y | categoría = k) = y_k / n_k
```

### 2.2 Smoothing

Para evitar overfitting en categorías con pocas muestras, se aplica smoothing hacia la media global:

```
encoded(k) = (n_k * mean_k + m * global_mean) / (n_k + m)
```

donde `m` es un parámetro de suavizado (típicamente 10–100).

### 2.3 Implementación con validación cruzada

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

class SmoothTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, m=30, cv_folds=5):
        self.m = m
        self.cv_folds = cv_folds

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        self.category_stats_ = {}
        for col in X.columns:
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            self.category_stats_[col] = stats
        return self

    def transform(self, X, y=None):
        X = X.copy()
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        encoded = np.zeros((len(X), len(X.columns)))

        if y is not None:
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                for i, col in enumerate(X.columns):
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    stats = y_train_fold.groupby(X_train_fold[col]).agg(['mean', 'count'])
                    global_mean = y_train_fold.mean()
                    X_val = X.iloc[val_idx][col]
                    n_k = X_val.map(lambda x: stats.loc[x, 'count'] if x in stats.index else 0)
                    mean_k = X_val.map(lambda x: stats.loc[x, 'mean'] if x in stats.index else global_mean)
                    encoded[val_idx, i] = (n_k * mean_k + self.m * global_mean) / (n_k + self.m)
        else:
            for i, col in enumerate(X.columns):
                stats = self.category_stats_[col]
                n_k = X[col].map(lambda x: stats.loc[x, 'count'] if x in stats.index else 0)
                mean_k = X[col].map(lambda x: stats.loc[x, 'mean'] if x in stats.index else self.global_mean_)
                encoded[:, i] = (n_k * mean_k + self.m * self.global_mean_) / (n_k + self.m)

        return encoded
```

**Salida esperada:** La transformación reemplaza cada valor categórico por un número entre 0 y 1 (para target binario) que representa la probabilidad suavizada de pertenecer a la clase positiva. En validación cruzada, cada fold usa estadísticas del fold de entrenamiento, eliminando leakage.

### 2.4 Cuándo usarlo

- Categorías con alta cardinalidad (códigos postales, IDs de producto, SKUs)
- Cuando one-hot encoding genera matrices demasiado dispersas
- En combinación con [[Regularization]] para controlar overfitting

---

## 3. Cyclical Encoding

Variables como hora del día, día de la semana, mes del año son cíclicas: 23:00 y 00:00 están cerca, pero codificadas linealmente (23 y 0) aparecen como opuestas.

### 3.1 Transformación seno/coseno

```python
import numpy as np
import pandas as pd

def cyclical_encode(df, col, period):
    """
    Transforma una variable cíclica en dos componentes seno/coseno.

    Args:
        df: DataFrame
        col: nombre de la columna
        period: período del ciclo (24 para hora, 7 para día, 12 para mes)

    Returns:
        DataFrame con dos columnas adicionales: {col}_sin, {col}_cos
    """
    df = df.copy()
    angles = 2 * np.pi * df[col] / period
    df[f'{col}_sin'] = np.sin(angles)
    df[f'{col}_cos'] = np.cos(angles)
    return df

# Ejemplo con hora del día
hours = pd.DataFrame({'hour': range(24)})
encoded = cyclical_encode(hours, 'hour', 24)
print(encoded.head())
```

**Salida esperada:**

```
   hour  hour_sin  hour_cos
0     0  0.000000  1.000000
1     1  0.258819  0.965926
2     2  0.500000  0.866025
3     3  0.707107  0.707107
4     4  0.866025  0.500000
```

La distancia euclidiana entre 23 y 0 es pequeña en el espacio seno/coseno, reflejando su cercanía real.

### 3.2 Escalamiento

Siempre escalar las features seno/coseno si el modelo lo requiere. Ambas componentes ya están en [-1, 1], pero otros modelos pueden necesitar [[Regularization]] o estandarización adicional cuando se combinan con otras features.

### 3.3 Consideraciones

- No eliminar la feature original sin probar: algunos modelos (árboles) pueden beneficiarse de tener ambos formatos
- Para variables con múltiples ciclos (ej. demanda tiene ciclo diario y semanal), codificar ambos períodos

---

## 4. Feature Interaction

Las interacciones capturan relaciones no lineales entre pares de features. Un modelo lineal `y = w1*x1 + w2*x2` no puede modelar que el efecto de `x1` depende del valor de `x2`.

### 4.1 PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

X = pd.DataFrame({
    'age': [25, 30, 45],
    'income': [40000, 60000, 80000]
})

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)
print(pd.DataFrame(X_poly, columns=feature_names))
```

**Salida esperada:**

```
    age   income   age^2   age income   income^2
0  25.0  40000.0  625.0  1000000.0  1.6e+09
1  30.0  60000.0  900.0  1800000.0  3.6e+09
2  45.0  80000.0  2025.0  3600000.0  6.4e+09
```

### 4.2 Interacciones con árboles

Los árboles de decisión capturan interacciones naturalmente, pero no explícitamente. Una técnica avanzada es entrenar un [[Ensemble Methods]] como Gradient Boosting, extraer las hojas donde caen las observaciones, y usar esas hojas como features de interacción para un modelo lineal.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Entrenar GBM
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3)
gbm.fit(X_train, y_train)

# Extraer leaf indices de cada árbol
leaf_indices = gbm.apply(X_train)  # shape: (n_samples, n_estimators)

# One-hot encode leaf indices → features de interacción
encoder = OneHotEncoder(sparse_output=True)
interaction_features = encoder.fit_transform(leaf_indices)
```

**Nota:** Las interacciones explícitas aumentan la dimensionalidad cuadráticamente. Siempre validar con un conjunto de hold-out.

---

## 5. Temporal Features

Cuando los datos tienen un componente temporal, las features tradicionales ignoran la estructura secuencial crítica para [[Time Series Fundamentals]].

### 5.1 Lag Features

```python
def create_lag_features(df, col, lags=[1, 7, 14, 28]):
    df = df.copy()
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df
```

### 5.2 Rolling Window Statistics

```python
def create_rolling_features(df, col, windows=[7, 14, 30]):
    df = df.copy()
    for w in windows:
        df[f'{col}_rolling_mean_{w}'] = df[col].rolling(window=w).mean()
        df[f'{col}_rolling_std_{w}'] = df[col].rolling(window=w).std()
        df[f'{col}_rolling_max_{w}'] = df[col].rolling(window=w).max()
        df[f'{col}_rolling_min_{w}'] = df[col].rolling(window=w).min()
    return df
```

### 5.3 Expanding Window

```python
def create_expanding_features(df, col):
    df = df.copy()
    df[f'{col}_expanding_mean'] = df[col].expanding().mean()
    df[f'{col}_expanding_std'] = df[col].expanding().std()
    df[f'{col}_expanding_count'] = df[col].expanding().count()
    return df
```

### 5.4 Time Since Event

```python
def time_since_last_event(df, event_col, date_col):
    """
    Calcula días/horas desde el último evento para cada observación.

    Args:
        df: DataFrame ordenado por fecha
        event_col: columna binaria (1 = evento ocurrió)
        date_col: columna de fecha/datetime

    Returns:
        Columna con tiempo desde el último evento
    """
    df = df.copy()
    last_event = None
    days_since = []

    for _, row in df.iterrows():
        if last_event is None:
            days_since.append(np.nan)
        else:
            days_since.append((row[date_col] - last_event).days)
        if row[event_col] == 1:
            last_event = row[date_col]

    df['days_since_last_event'] = days_since
    return df
```

**Precaución:** Al crear lag y rolling features, las primeras filas tendrán NaN. Decidir si llenar con 0, con la media, o eliminarlas depende del contexto.

---

## 6. Automated Feature Engineering

La ingeniería manual no escala a docenas de tablas relacionales. Herramientas como Featuretools automatizan la creación de features a partir de [[Relational Model & SQL Fundamentals]].

### 6.1 Deep Feature Synthesis (DFS)

```python
import featuretools as ft

# Definir entidades y relaciones
customers = ft.Entity(id='customers')
sessions = ft.Entity(id='sessions')
transactions = ft.Entity(id='transactions')

# Relaciones: customers → sessions → transactions
relationships = [
    ft.Relationship(customers['customer_id'], sessions['customer_id']),
    ft.Relationship(sessions['session_id'], transactions['session_id'])
]

# Stack de entidades
es = ft.EntitySet('ecommerce')
es = es.add_dataframe(dataframe_name='customers', dataframe=customers_df, index='customer_id')
es = es.add_dataframe(dataframe_name='sessions', dataframe=sessions_df, index='session_id')
es = es.add_dataframe(dataframe_name='transactions', dataframe=transactions_df, index='transaction_id')
es = es.add_relationship(relationships[0])
es = es.add_relationship(relationships[1])

# Generar features automáticamente
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='customers',
    max_depth=2,
    agg_primitives=['sum', 'mean', 'count', 'std', 'max', 'min'],
    trans_primitives=['day', 'month', 'weekday', 'hour']
)
```

**Salida esperada:** Un DataFrame con cientos de features generadas automáticamente, como `MEAN(sessions.transactions.amount)`, `COUNT(transactions)`, `STD(sessions.duration)`.

### 6.2 Ventajas y desventajas

- Ventaja: escala a decenas de tablas, cientos de columnas, relaciones 1:N y N:M
- Desventaja: genera cientos de features irrelevantes que requieren [[Feature Engineering]] de selección posterior
- Recomendación: usar DFS para generar candidatas y luego aplicar selección rigurosa

---

## 7. Feature Selection

Más features no es mejor. Seleccionar las features correctas reduce overfitting, mejora interpretabilidad y acelera entrenamiento.

### 7.1 Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# F-test
selector_f = SelectKBest(score_func=f_classif, k=20)
X_selected_f = selector_f.fit_transform(X, y)

# Mutual information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=20)
X_selected_mi = selector_mi.fit_transform(X, y)

print("F-test scores:", selector_f.scores_[:5])
print("MI scores:", selector_mi.scores_[:5])
```

### 7.2 Wrapper Methods — Boruta

Boruta entrena un Random Forest y compara la importancia de cada feature contra copias aleatorias (shadow features). Features significativamente mejores que su shadow se mantienen.

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta = BorutaPy(rf, n_estimators='auto', random_state=42, max_iter=100)
boruta.fit(X.values, y.values)

# Features confirmadas
selected_features = X.columns[boruta.support_].tolist()
print("Features confirmadas:", selected_features)
```

### 7.3 Embedded Methods — Permutation Importance

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier().fit(X_train, y_train)
result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)

feature_importance = pd.DataFrame({
    'feature': X_val.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

**Salida esperada:** Tabla de features ordenadas por importancia, con desviación estándar que indica estabilidad del ranking.

---

## 8. Common Mistakes

### Target encoding con data leakage
- **Error:** Calcular las medias del target usando todo el dataset y luego aplicarlas al mismo dataset
- **Solución:** Siempre usar validación cruzada para calcular target encoding dentro de cada fold, como en el código de la Sección 2

### Cyclical encoding sin escalar
- **Error:** Usar seno/coseno pero no escalar el resto de las features, causando que el modelo ignore las componentes cíclicas
- **Solución:** Escalar todas las features al mismo rango (por ej., StandardScaler)

### Interacciones sin validación
- **Error:** Generar PolynomialFeatures de grado 3 o 4 sin regularización ni selección, resultando en dimensionalidad explosiva
- **Solución:** Limitar grado máximo, usar [[Regularization]] (Lasso, Ridge), o solo interacciones de pares

### Lag features con leakage temporal
- **Error:** Usar información futura (shift negativo o lag = 0) para predecir el presente
- **Solución:** Asegurar que shift >= 1 y que el train-test split respeta el orden temporal

### Automated Feature Engineering sin filtro
- **Error:** Usar DFS y meter las 500 features generadas directamente al modelo
- **Solución:** Aplicar selección de features (Boruta, permutation importance) post-DFS

---

## Resumen

| Técnica | Tipo de dato | Riesgo principal |
|---|---|---|
| Target Encoding | Categórico alta cardinalidad | Data leakage |
| Cyclical Encoding | Temporal cíclico | Subescalamiento |
| Polynomial Interactions | Numérico | Dimensionalidad |
| Lag/Rolling Features | Serie temporal | Leakage temporal |
| DFS | Relacional (múltiples tablas) | Features irrelevantes |

El feature engineering avanzado no es magia: es aplicar la transformación correcta al tipo de dato correcto con validación rigurosa. Mide el impacto de cada grupo de features por separado para justificar su inclusión.

---

## Check Your Understanding

**1. ¿Por qué target encoding necesita validación cruzada?**
<!-- Porque calcular la media del target sobre todo el dataset introduce data leakage: el modelo "ve" el target en entrenamiento. CV calcula las medias solo con datos del fold de entrenamiento. -->

**2. ¿Cuál es la ventaja de seno/coseno sobre codificar hora como 0-23?**
<!-- Preserva la distancia cíclica: 23:00 y 00:00 están cerca en el espacio seno/coseno, mientras que 23 y 0 son extremos opuestos en codificación lineal. -->

**3. ¿Por qué PolynomialFeatures con degree=3 en 10 features originales es problemático?**
<!-- El número de features crece como O(n^d). degree=3 con 10 features genera ~286 términos, incluyendo combinaciones que pueden causar overfitting si no hay suficiente regularización. -->

**4. ¿Cuándo preferirías Boruta a SelectKBest?**
<!-- Boruta considera interacciones entre features (usa Random Forest) y tiene un criterio estadístico para decidir si una feature es relevante. SelectKBest evalúa cada feature independientemente. -->

**5. ¿Qué precaución tomar al crear lag features para series temporales?**
<!-- 1) No usar lag=0 (información del mismo instante), 2) Asegurar que train/test split respeta el orden cronológico, 3) Decidir cómo manejar NaN de las primeras observaciones. -->

---

## Where to Go Next

- [[Feature Engineering]] — Conceptos fundamentales antes de las técnicas avanzadas
- [[Supervised Learning]] — Cómo evaluar el impacto de las features en el modelo
- [[Ensemble Methods]] — Modelos que capturan interacciones automáticamente
- [[Regularization]] — Controlar overfitting cuando agregamos muchas features
- [[Time Series Fundamentals]] — Features temporales en contexto de series de tiempo
- [[Data Pipelines & ETL]] — Cómo operationalizar feature engineering en producción
- [[Relational Model & SQL Fundamentals]] — Base para entender feature engineering relacional con DFS
