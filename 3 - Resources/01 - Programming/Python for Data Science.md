---
tags: [programming, python, data-science, applied]
status: growing
created: 2026-06-27
---

# Python para Ciencia de Datos

## 1. Escenario de aprendizaje

Te entregan un conjunto de datos de 50 GB en formato CSV. Necesitas cargarlo, limpiarlo, transformarlo y entrenar un modelo de clasificación. Si usas Python puro con bucles, el script tardará horas — o se quedará sin memoria. Necesitas NumPy para operaciones vectorizadas rápidas, pandas para manipulación eficiente de datos y scikit-learn para el pipeline de ML. Entender estas bibliotecas en profundidad (no solo cómo usarlas, sino **cómo funcionan internamente**) te permite escribir código de datos más rápido, correcto y mantenible.

Los fundamentos de Python son necesarios pero no suficientes. El ecosistema de ciencia de datos — NumPy, pandas, scikit-learn, PyTorch — es lo que hace de Python el lenguaje dominante para ML.

---

## 2. NumPy — La base

NumPy proporciona el **ndarray** (array n-dimensional), que es más rápido y eficiente en memoria que las listas de Python para datos numéricos.

### 2.1 Por qué NumPy es rápido

Las listas de Python contienen punteros a objetos — cada elemento es un objeto Python. Los arrays de NumPy almacenan **valores C puros** en memoria contigua.

```python
# Lista de Python: cada elemento es un puntero PyObject
py_list = [1, 2, 3]          # ~72 bytes por elemento

# Array de NumPy: valores int64 puros en un bloque contiguo
np_array = np.array([1, 2, 3])  # 8 bytes por elemento + overhead
```

Esto significa:
- **Menos memoria**: 8 bytes por int vs ~72 bytes
- **Amigable con la caché de CPU**: el acceso secuencial a memoria es más rápido
- **Operaciones vectorizadas**: las operaciones se ejecutan en C, no en bucles de Python

### 2.2 Creación de arrays

```python
import numpy as np

arr = np.array([1, 2, 3])
zeros = np.zeros((3, 4))           # 3 filas, 4 columnas
ones = np.ones((2, 3))
eye = np.eye(3)                    # matriz identidad (3×3)
linspace = np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1.0]
random = np.random.randn(1000)     # 1000 muestras de N(0, 1)
```

### 2.3 Vectorización — El concepto clave

**Sin vectorización** (lento):
```python
result = np.zeros(1000)
for i in range(1000):
    result[i] = a[i] * b[i] + c[i]  # Bucle de Python — interpretado, lento
```

**Con vectorización** (rápido):
```python
result = a * b + c                   # Sin bucle de Python — se ejecuta en C
```

**Regla**: nunca iteres sobre arrays de NumPy en Python. Usa operaciones vectorizadas. [[Data Structures & Algorithms]] explica los conceptos de Big O detrás de esta ventaja de rendimiento.

### 2.4 Broadcasting

Las operaciones entre arrays de diferentes formas se alinean automáticamente:

```python
matrix = np.ones((3, 4))      # 3×4
row_mean = matrix.mean(axis=1)  # forma (3,) — media de cada fila
centered = matrix - row_mean[:, np.newaxis]  # broadcasting a través de columnas

# Reglas de broadcasting:
# (3, 4) and (3,) → insertar dimensión → (3, 4) and (3, 1) → (3, 4)
```

### 2.5 Álgebra Lineal

```python
# Multiplicación de matrices
C = A @ B                      # igual que np.matmul(A, B)
v = W @ x + b                  # capa de red neuronal

# Descomposiciones
eigvals, eigvecs = np.linalg.eig(cov_matrix)
U, S, Vt = np.linalg.svd(matrix)
```

---

## 3. pandas — Manipulación de datos

pandas está construido sobre NumPy y añade filas y columnas etiquetadas.

### 3.1 El DataFrame

Un DataFrame es conceptualmente un **diccionario de Series** (columnas). Cada columna tiene un solo tipo.

```python
import pandas as pd

df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json")

# Inspección básica
df.info()          # columnas, tipos, conteos no nulos
df.describe()      # estadísticas resumidas
df.head()          # primeras 5 filas
df.shape           # (filas, columnas)
```

### 3.2 Selección — .loc vs .iloc

Esta es la fuente más común de confusión.

```python
# .loc: basado en etiquetas (nombres de columna, índices)
df.loc[rows, columns]
df.loc[df["age"] > 30, ["name", "income"]]  # máscara booleana de filas

# .iloc: basado en posición entera
df.iloc[row_indices, column_indices]
df.iloc[10:20, [0, 3, 5]]  # filas 10-19, columnas 0, 3, 5
```

**Cuándo usar cada uno**: `.loc` cuando conoces los nombres de las columnas (la mayoría del tiempo), `.iloc` cuando iterás programáticamente.

### 3.3 El patrón Split-Apply-Combine

Este es el patrón de datos más importante en pandas — refleja el paradigma map-filter-reduce de [[Functional Programming]]:

```python
# Agrupar por categoría → aplicar función → combinar resultados
df.groupby("department")["salary"].agg(["mean", "std", "count"])

# Paso a paso:
# 1. Split: dividir los datos por departamento
# 2. Apply: calcular media, std, conteo para cada uno
# 3. Combine: fusionar los resultados en un nuevo DataFrame

# Más complejo
result = (df
    .groupby(["department", "year"])
    .agg(
        avg_salary=("salary", "mean"),
        headcount=("employee_id", "nunique"),
        total_bonus=("bonus", "sum"),
    )
    .reset_index())
```

### 3.4 Fusiones y Joins

```python
# Joins estilo SQL
pd.merge(orders, customers, on="customer_id", how="inner")
pd.merge(orders, customers, on="customer_id", how="left")

# Concatenación
pd.concat([df1, df2], axis=0)  # apilar filas (crece el largo)
pd.concat([df1, df2], axis=1)  # lado a lado (crecen las columnas)
```

### 3.5 Consejos de rendimiento

- **Evita `apply` con funciones de Python** para conjuntos de datos grandes (itera en Python). Usa operaciones vectorizadas en su lugar.
- **Usa el tipo categorical** para columnas de texto con baja cardinalidad (ahorra memoria, groupbys más rápidos).
- **Usa inplace=False** (el valor por defecto) — es más seguro y a menudo igual de rápido.
- **`query()` es más legible** para filtros complejos: `df.query("age > 30 and income > 50000")`

---

## 4. scikit-learn — La interfaz de ML

scikit-learn estableció la API consistente `fit` / `predict` / `transform` que se convirtió en estándar en ML en Python.

### 4.1 El patrón de la API

Cada modelo y transformador sigue el mismo patrón, una aplicación clásica del polimorfismo de [[Object-Oriented Programming]]:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100)),
])

pipeline.fit(X_train, y_train)        # aprender parámetros
predictions = pipeline.predict(X_test)  # aplicar el modelo aprendido
```

### 4.2 Por qué importa la API uniforme

- **Componentes intercambiables**: cambia `RandomForestClassifier` por `XGBClassifier` sin cambiar nada más
- **Búsqueda de hiperparámetros**: `GridSearchCV` funciona en cualquier modelo con la misma API
- **Pipelines**: compone preprocesamiento + modelado en un solo objeto
- **Producción**: los exportadores y frameworks de servicio esperan esta API

### 4.3 Herramientas esenciales

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

## 5. PyTorch vs TensorFlow — Una orientación rápida

Ambos son frameworks de deep learning. PyTorch se ha convertido en la opción dominante para investigación y cada vez más para producción.

**Filosofía clave de PyTorch**: define-by-run (grafos de cómputo dinámicos).

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
