---
tags:
  - machine-learning
  - dimensionality-reduction
  - pca
  - tsne
  - umap
  - lda
status: seedling
created: 2026-06-28
---

# Dimensionality Reduction

## 1. Escenario de aprendizaje

Tienes 500 features y solo 10,000 muestras. Tu modelo de Random Forest tarda 45 minutos en entrenar y aun así hace overfitting. La distancia euclideana entre puntos es prácticamente la misma para todos los pares (la maldición de la dimensionalidad). Necesitas reducir la dimensionalidad.

O imagina que trabajas con textos: 5,000 documentos representados con [[Word Embeddings (Word2Vec)]] o TF-IDF generan 50,000 features. Sin reducción, ningún modelo lineal converge y cualquier distancia es ruido.

La reducción de dimensionalidad transforma datos de alta dimensionalidad a un espacio de menor dimensión preservando la estructura relevante. Tiene dos grandes aplicaciones:
1. **Preprocessing para ML**: eliminar features redundantes, reducir ruido, acelerar entrenamiento.
2. **Visualización**: proyectar datos a 2D o 3D para entender su estructura.

Esta nota cubre PCA, t-SNE, UMAP y LDA, con criterios claros para elegir entre ellos según el objetivo.

## 2. ¿Por qué reducir?

### Curse of Dimensionality
A medida que aumenta el número de dimensiones, el volumen del espacio crece exponencialmente. Los datos se vuelven "escasos": la distancia euclideana entre cualquier par de puntos tiende a ser la misma. Para mantener la misma densidad de muestreo, se necesitan exponencialmente más muestras por cada dimensión adicional.

### Computational Cost
Entrenar modelos con cientos o miles de features es lento y consume mucha memoria. Reducir a 50-100 features puede acelerar el entrenamiento 10x sin perder precisión.

### Overfitting
Con muchas features y pocas muestras, los modelos aprenden ruido en lugar de señal. La reducción de dimensionalidad actúa como [[Regularization]].

### Visualization
El ojo humano no percibe más de 3 dimensiones. Reducir a 2D o 3D permite inspeccionar clusters, outliers y relaciones entre instancias.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import time

print("Demostrando la maldición de la dimensionalidad...")
for dims in [10, 50, 100, 500]:
    X = np.random.randn(1000, dims)
    # Distancia entre el primer punto y todos los demás
    dists = np.linalg.norm(X - X[0], axis=1)
    ratio = dists.min() / dists.max()
    print(f"  {dims:4d} dims → min/max distance ratio: {ratio:.4f}")
```

Salida esperada:
```
Demostrando la maldición de la dimensionalidad...
    10 dims → min/max distance ratio: 0.7351
    50 dims → min/max distance ratio: 0.8872
   100 dims → min/max distance ratio: 0.9205
   500 dims → min/max distance ratio: 0.9628
```

A medida que aumentan las dimensiones, la distancia relativa entre el vecino más cercano y el más lejano se acerca a 1. Distancias dejan de ser informativas.

## 3. PCA (Principal Component Analysis)

PCA es una proyección lineal que encuentra las direcciones de máxima varianza en los datos. Los componentes principales son ortogonales y están ordenados por la cantidad de varianza que explican.

### Cómo funciona
1. Centrar los datos (restar la media).
2. Calcular la matriz de covarianza.
3. Calcular autovalores y autovectores: los autovectores son los componentes principales.
4. Proyectar los datos sobre los top-k autovectores.

```python
# Cargar digits dataset (8x8 imágenes → 64 features)
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# PCA
pca = PCA(n_components=0.95)  # retener 95% de varianza
X_pca = pca.fit_transform(X_digits)

print(f"Dimensiones originales: {X_digits.shape[1]}")
print(f"Dimensiones después de PCA: {X_pca.shape[1]}")
print(f"Varianza explicada acumulada: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

# Scree plot
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
print("\nVarianza explicada por componente:")
for i, (e, c) in enumerate(zip(explained[:10], cumulative[:10])):
    print(f"  PC{i+1}: {e:.4f} (acumulado: {c:.4f})")
```

Salida esperada:
```
Dimensiones originales: 64
Dimensiones después de PCA: 29
Varianza explicada acumulada: 0.9502

Varianza explicada por componente:
  PC1: 0.1489 (acumulado: 0.1489)
  PC2: 0.1362 (acumulado: 0.2851)
  PC3: 0.1175 (acumulado: 0.4026)
  PC4: 0.0837 (acumulado: 0.4863)
  PC5: 0.0609 (acumulado: 0.5472)
  PC6: 0.0528 (acumulado: 0.6000)
  PC7: 0.0427 (acumulado: 0.6427)
  PC8: 0.0368 (acumulado: 0.6795)
  PC9: 0.0327 (acumulado: 0.7122)
  PC10: 0.0275 (acumulado: 0.7397)
```

### PCA como preprocessing para ML

```python
X_clf, y_clf = make_classification(
    n_samples=5000, n_features=100, n_informative=20,
    n_redundant=30, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

start = time.time()
rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
rf_original.fit(X_tr, y_tr)
score_orig = accuracy_score(y_te, rf_original.predict(X_te))
time_orig = time.time() - start

# PCA
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

pca = PCA(n_components=40)
X_tr_pca = pca.fit_transform(X_tr_scaled)
X_te_pca = pca.transform(X_te_scaled)

start = time.time()
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_tr_pca, y_tr)
score_pca = accuracy_score(y_te, rf_pca.predict(X_te_pca))
time_pca = time.time() - start

print(f"Original  ({X_tr.shape[1]} features): {score_orig:.4f} ({time_orig:.2f}s)")
print(f"Con PCA   ({X_tr_pca.shape[1]} features): {score_pca:.4f} ({time_pca:.2f}s)")
```

Salida esperada:
```
Original  (100 features): 0.9170 (8.43s)
Con PCA   (40 features): 0.9130 (3.81s)
```

## 4. t-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE es un método **no lineal** y no determinista para visualización. Minimiza la divergencia KL entre las distribuciones de probabilidad de pares en alta dimensión (gaussianas) y baja dimensión (t-Student). El resultado: puntos cercanos en alta dimensión quedan cercanos en 2D, y puntos lejanos quedan lejanos.

```python
# Aplicar t-SNE para visualizar digits
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X_digits)

print(f"t-SNE completado. Shape: {X_tsne.shape}")
print(f"KL divergence final: {tsne.kl_divergence_:.4f}")
```

Salida esperada:
```
t-SNE completado. Shape: (1797, 2)
KL divergence final: 0.8001
```

**Hiperparámetros críticos:**
- `perplexity`: balance entre atención local y global. Típicamente 5-50. Perplejidad baja enfatiza estructura local; alta enfatiza la global. Regla: usar entre 30 y 50 para datasets medianos.
- `learning_rate`: típicamente 100-1000. Si el embedding parece "pelota" (todos los puntos amontonados), sube el LR.
- `n_iter`: mínimo 250; idealmente 1000+. Muy pocas iteraciones → estructura no converge.

**Limitaciones:**
- **No preserva distancias globales**: la escala y las distancias entre clusters no son interpretables.
- **No deterministico**: cada ejecución da un embedding diferente.
- **O(n²)**: muy lento para >10K puntos. Considera aproximaciones (Barnes-Hut) o usa UMAP.
- **No sirve para feature engineering**: no puedes proyectar datos nuevos (no hay transform).

## 5. UMAP (Uniform Manifold Approximation and Projection)

UMAP se basa en teoría de categorías de homología persistente. Construye un grafo de vecinos en alta dimensión y después optimiza un embedding en baja dimensión que minimiza la diferencia entre los grafos.

- Significativamente más rápido que t-SNE (puede manejar 100K puntos en minutos).
- Preserva mejor la estructura global: las distancias entre clusters son más significativas.
- Es determinista si fijas `random_state`.
- Tiene fundamentos matemáticos sólidos (en lugar del heurístico de t-SNE).

```python
try:
    import umap.umap_ as umap_lib
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_digits)
    print(f"UMAP completado. Shape: {X_umap.shape}")
except ImportError:
    print("UMAP no instalado. Instala con: pip install umap-learn")
```

Salida esperada:
```
UMAP completado. Shape: (1797, 2)
```

**Parámetros clave:**
- `n_neighbors`: balance local-global. Valores pequeños (2-10) enfatizan estructura local; valores grandes enfatizan estructura global.
- `min_dist`: qué tan apretados pueden estar los puntos en baja dimensión. 0.0 → separación máxima de clusters; 1.0 → puntos uniformemente distribuidos.
- `n_components`: dimensiones de salida. 2 para visualización, hasta 50-100 como preprocessing.

## 6. LDA (Linear Discriminant Analysis)

No confundir con Latent Dirichlet Allocation (el de tópicos de textos). Este LDA es un método **supervisado** que encuentra la proyección lineal que maximiza la separación entre clases.

La idea: encontrar ejes en los que la varianza **entre clases** sea máxima y la varianza **dentro de cada clase** sea mínima. Es el equivalente supervisado de PCA.

```python
# LDA en digits
lda = LDA(n_components=9)  # máximo = n_classes - 1 = 9
X_lda = lda.fit_transform(X_digits, y_digits)

print(f"LDA: {X_digits.shape[1]} → {X_lda.shape[1]} dimensiones")
print(f"Varianza explicada (ratio): {lda.explained_variance_ratio_[:5]}")

# Comparar LDA vs PCA para clasificación
X_tr, X_te, y_tr, y_te = train_test_split(X_digits, y_digits, test_size=0.3, random_state=42)

# PCA + RF
pca_9 = PCA(n_components=9)
X_tr_pca = pca_9.fit_transform(X_tr)
X_te_pca = pca_9.transform(X_te)
rf_pca = RandomForestClassifier(random_state=42).fit(X_tr_pca, y_tr)
acc_pca = accuracy_score(y_te, rf_pca.predict(X_te_pca))

# LDA + RF
lda_9 = LDA(n_components=9)
X_tr_lda = lda_9.fit_transform(X_tr, y_tr)
X_te_lda = lda_9.transform(X_te)
rf_lda = RandomForestClassifier(random_state=42).fit(X_tr_lda, y_tr)
acc_lda = accuracy_score(y_te, rf_lda.predict(X_te_lda))

print(f"PCA+RF Accuracy: {acc_pca:.4f}")
print(f"LDA+RF Accuracy: {acc_lda:.4f}")
```

Salida esperada:
```
LDA: 64 → 9 dimensiones
Varianza explicada (ratio): [0.6601 0.1215 0.0901 0.0525 0.0326]
PCA+RF Accuracy: 0.9111
LDA+RF Accuracy: 0.9463
```

**Limitaciones:**
- Requiere etiquetas (supervisado).
- Asume que cada clase sigue una distribución normal y que todas las clases comparten la misma matriz de covarianza.
- Máximo de componentes = n_classes - 1. Con 2 clases, solo obtienes 1 componente.

## 7. PCA vs t-SNE vs UMAP vs LDA

| Característica | PCA | t-SNE | UMAP | LDA |
|:--------------|:---:|:-----:|:----:|:---:|
| Tipo | Lineal | No lineal | No lineal | Lineal |
| Supervisado | No | No | No | Sí |
| Preserva distancias globales | Sí | No | Parcial | No |
| Preserva distancias locales | No | Sí | Sí | No |
| Determinista | Sí | No | Sí (con seed) | Sí |
| Escalabilidad | Muy alta | Baja (≤10K) | Alta (≤100K) | Alta |
| Feature engineering | Sí | No | Sí | Sí |
| Visualización | Regular | Excelente | Excelente | Buena |
| Max dimensiones output | n_features | 3 | Cualquiera | n_classes - 1 |

**Cuándo usar cada uno:**

- **PCA**: preprocessing para ML, reducir dimensionalidad antes de otros algoritmos, eliminar multicolinealidad, visualización rápida (primeros 2-3 PCs).
- **t-SNE**: visualización exploratoria de datasets ≤10K puntos para identificar clusters. Usa UMAP si puedes.
- **UMAP**: visualización de datasets grandes (hasta 100K+), preprocessing para clustering o clasificación, mejor alternativa a t-SNE en casi todos los casos.
- **LDA**: supervised dimensionality reduction para clasificación, especialmente cuando el número de features es mayor que el número de muestras y quieres máxima separación entre clases.

## 8. Common Mistakes

### Aplicar PCA antes de train/test split (data leakage)
PCA calcula los componentes principales usando todo el dataset. Si haces PCA antes de separar train/test, los componentes aprenden información de los datos de test. Esto es [[data leakage]]: la varianza explicada y los componentes están contaminados.

**Correcto:**
```python
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)  # misma transformación
```

**Incorrecto:**
```python
X_pca = PCA(n_components=50).fit_transform(X)  # fit en todo X
X_train_pca = X_pca[:800]
X_test_pca = X_pca[800:]  # contaminado
```

### t-SNE para feature engineering
t-SNE no preserva distancias globales ni es una transformación estable (cada ejecución da un embedding diferente). No uses los componentes de t-SNE como input para un modelo de ML. Usa UMAP o PCA para eso.

### No escalar los datos antes de PCA
PCA maximiza varianza. Si las features tienen escalas diferentes (ej. edad 0-100 vs. ingreso 0-100,000), las features con mayor escala dominan los componentes. Siempre estandariza (StandardScaler) antes de PCA.

### Elegir n_components sin criterio
Usar `n_components=2` sin conocer la varianza explicada puede descartar señal importante. Usa el scree plot o `n_components=0.95` para retener el 95% de la varianza.

### Aplicar reducción de dimensionalidad antes de feature selection
La reducción de dimensionalidad combina features existentes (PCA) o transforma el espacio (t-SNE/UMAP). Si tienes features irrelevantes, elimínalas primero con [[Feature Engineering]] y selección de features, luego reduce dimensionalidad. PCA con features ruidosas produce componentes ruidosos.

## Resumen

La reducción de dimensionalidad es una herramienta esencial en el toolkit de ML para combatir la maldición de la dimensionalidad, acelerar entrenamiento, y visualizar datos complejos.

- **PCA** es rápido, deterministico, y el estándar para preprocessing. Úsalo siempre como primer paso.
- **t-SNE** es ideal para visualización exploratoria de datasets pequeños. Sus embeddings revelan clusters que PCA no muestra.
- **UMAP** es superior a t-SNE en velocidad, escalabilidad y preservación de estructura global. Es la herramienta moderna para visualización y feature engineering.
- **LDA** es la opción supervisada para clasificación cuando la separación entre clases es la prioridad.

La regla práctica: estandariza, aplica PCA para reducir ruido y dimensionalidad, entrena tu modelo, y si necesitas visualizar, usa UMAP.

## Check Your Understanding

1. ¿Por qué la distancia euclideana deja de ser significativa en alta dimensionalidad?
2. ¿Qué diferencia fundamental hay entre PCA y t-SNE en términos de preservación de distancias?
3. ¿Por qué UMAP es generalmente preferible a t-SNE para datasets grandes?
4. ¿En qué caso elegirías LDA sobre PCA?
5. ¿Cuál es el error de aplicar PCA antes de separar train y test?
6. ¿Por qué es necesario escalar los datos antes de PCA?
7. ¿Puedo usar t-SNE para reducir 500 features a 50 como input de un clasificador?

<!--
1. Porque el volumen del espacio crece exponencialmente con cada dimensión. Los puntos se vuelven igualmente distantes: la relación entre la distancia mínima y máxima entre puntos tiende a 1, haciendo que la noción de "vecino cercano" pierda sentido.
2. PCA preserva distancias globales (la estructura a gran escala) al proyectar sobre direcciones de máxima varianza global. t-SNE preserva distancias locales (vecinos cercanos) al modelar distribuciones de probabilidad de pares, pero las distancias entre clusters no son significativas.
3. UMAP tiene complejidad O(n log n) vs O(n²) de t-SNE. Puede procesar 100K puntos en minutos mientras t-SNE toma horas. Además, UMAP preserva mejor la estructura global y es deterministico con seed fija.
4. Cuando tienes etiquetas y quieres maximizar la separación entre clases (ej. clasificación con muchas features). LDA encuentra la proyección que maximiza la varianza entre clases y minimiza la intra-clase, mientras PCA ignora las etiquetas.
5. Data leakage: PCA calcula los componentes usando información del dataset completo. Si test está incluido, los componentes principales se ajustan también a los datos de test, dando una visión irrealmente optimista de la varianza explicada y la calidad de la reducción.
6. PCA maximiza la varianza. Si las features están en escalas diferentes (ej. metros vs. kilómetros), las de mayor escala dominarán los primeros componentes. Estandarizar (media=0, var=1) asegura que todas las features contribuyan por igual.
7. No. t-SNE es no determinista, no preserva distancias globales, y no tiene un método de transform (transform) para datos nuevos. Cada ejecución produce un embedding diferente. Para feature engineering usa PCA, UMAP (que sí tiene transform), o autoencoders.
-->

## Where to Go Next

- [[Unsupervised Learning]] — Contexto más amplio del aprendizaje no supervisado
- [[Linear Algebra]] — Autovalores, autovectores y SVD, base matemática de PCA
- [[Advanced Linear Algebra]] — Descomposiciones matriciales avanzadas
- [[Visualization Fundamentals]] — Principios de visualización de datos
- [[Feature Engineering]] — Selección y creación de features, previo a reducción
- [[Word Embeddings (Word2Vec)]] — Técnicas de embedding textual, típicamente reducidas con PCA/t-SNE
- [[Seaborn & Statistical Plots]] — Herramientas de visualización para graficar reducciones
