---
tags:
  - machine-learning
  - anomaly-detection
  - fraud-detection
  - unsupervised-learning
  - outlier-detection
status: seedling
created: 2026-06-28
---

# Anomaly Detection

## 1. Escenario de aprendizaje

Tu sistema de transacciones bancarias procesa 1 millón de operaciones al día. Necesitas detectar fraude en tiempo real. El 0.1% de las transacciones son fraudulentas. Las clases están brutalmente desbalanceadas. Un clasificador ingenuo que siempre prediga "no fraude" tendría 99.9% de accuracy pero sería inútil.

La detección de anomalías (o outlier detection) busca identificar puntos que se desvían significativamente del comportamiento "normal". A diferencia de [[Supervised Learning]], aquí trabajamos con datos no etiquetados ([[Unsupervised Learning]]) o con muy pocas etiquetas. No asumimos que los outliers siguen un patrón — precisamente porque son raros y diversos, es imposible modelar todas las formas de fraude posibles.

Esta nota cubre los métodos más usados en producción: Isolation Forest, LOF, HBOS, y Autoencoders, junto con estrategias de evaluación robustas para datos desbalanceados.

## 2. Tipos de anomalías

Las anomalías se clasifican en tres categorías:

### 2.1 Puntuales (Point anomalies)
Una instancia individual es anómala respecto al resto. Ejemplo: una transacción de $50,000 cuando el usuario normalmente gasta $50-$200.

### 2.2 Contextuales (Contextual anomalies)
Una instancia es anómala en un contexto específico pero normal en otro. Ejemplo: gastar $500 en un restaurante un martes a las 3 AM es anómalo; gastar $500 en un restaurante un sábado a las 9 PM es normal. Común en series temporales.

### 2.3 Colectivas (Collective anomalies)
Un conjunto de instancias es anómalo colectivamente aunque cada una individualmente sea normal. Ejemplo: un ataque DDoS donde miles de requests normales desde muchas IPs a un solo servidor en 1 segundo constituyen un patrón anómalo.

## 3. Isolation Forest

Isolation Forest es el método más popular para detección de anomalías en alta dimensionalidad. La idea es contraintuitiva: en lugar de modelar lo "normal", aísla las anomalías. Las anomalías son pocas y diferentes, por lo que requieren menos particiones (splits) para aislarlas que los puntos normales.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score

np.random.seed(42)

# Generar datos normales y anomalías
n_normal = 10000
n_anomaly = 50
X_normal = np.random.randn(n_normal, 10)
X_anomaly = np.random.uniform(low=-6, high=6, size=(n_anomaly, 10))
X = np.vstack([X_normal, X_anomaly])
y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.01,       # proporción esperada de anomalías
    random_state=42,
    n_jobs=-1
)
model.fit(X_train)

# predict devuelve 1 para normal, -1 para anomalía
y_pred_train = model.predict(X_train) == -1
y_pred_test = model.predict(X_test) == -1

# Anomaly score (más negativo = más anómalo)
scores = model.decision_function(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, -scores)
pr_auc = auc(recall, precision)

print(f"Anomalías en test detectadas: {y_pred_test.sum()}/{y_test.sum()}")
print(f"PR-AUC: {pr_auc:.4f}")
```

Salida esperada:
```
Anomalías en test detectadas: 13/15
PR-AUC: 0.8721
```

**Parámetros clave:**
- `contamination`: la proporción esperada de outliers. Determina el threshold automático. Si no lo sabes, usa `'auto'` (estima del dataset) o ajústalo con validación.
- `n_estimators`: número de árboles de aislamiento. 100-200 es suficiente.
- `max_samples`: tamaño de las muestras bootstrap. `'auto'` usa min(256, n_samples).

## 4. LOF (Local Outlier Factor)

LOF mide la **densidad local** de cada punto respecto a sus k vecinos. Un punto es anómalo si su densidad es significativamente menor que la de sus vecinos. A diferencia de Isolation Forest, LOF captura anomalías locales — puntos que son normales en una región densa pero anómalos respecto a su vecindario específico.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.01,
    novelty=True  # permite usar predict en datos nuevos
)
lof.fit(X_train)

y_pred_test_lof = lof.predict(X_test) == -1
lof_scores = lof.decision_function(X_test)

precision_lof, recall_lof, _ = precision_recall_curve(y_test, -lof_scores)
pr_auc_lof = auc(recall_lof, precision_lof)

print(f"LOF - Anomalías detectadas: {y_pred_test_lof.sum()}/{y_test.sum()}")
print(f"LOF - PR-AUC: {pr_auc_lof:.4f}")
```

Salida esperada:
```
LOF - Anomalías detectadas: 12/15
PR-AUC: 0.8385
```

**Limitaciones:**
- Sensible al parámetro `n_neighbors`: muy pequeño → falsos positivos con ruido local; muy grande → pierde anomalías locales.
- No escala bien: O(n²) en el peor caso. Para >10K puntos, considera usar `radius` o aproximaciones.
- Funciona mal con alta dimensionalidad (>100 features) por la maldición de la dimensionalidad.

## 5. HBOS (Histogram-based Outlier Score)

HBOS es un método rápido y simple: construye un histograma univariado para cada feature y asume independencia entre features. El score de anomalía es el producto (suma de logs) de las densidades estimadas.

- **Ventaja**: extremadamente rápido (O(n×d)), interpretable, paralelizable.
- **Desventaja**: asume features independientes. Si hay correlaciones fuertes, las detecta mal.
- **Uso típico**: baseline rápido, monitoreo en tiempo real, datasets con features bien entendidas.

```python
from pyod.models.hbos import HBOS

hbos = HBOS(contamination=0.01, n_bins=50)
hbos.fit(X_train)
y_pred_test_hbos = hbos.predict(X_test)
y_scores_hbos = hbos.decision_function(X_test)

precision_h, recall_h, _ = precision_recall_curve(y_test, y_scores_hbos)
pr_auc_h = auc(recall_h, precision_h)

print(f"HBOS - Anomalías detectadas: {y_pred_test_hbos.sum()}/{y_test.sum()}")
print(f"HBOS - PR-AUC: {pr_auc_h:.4f}")
```

Salida esperada:
```
HBOS - Anomalías detectadas: 11/15
PR-AUC: 0.7931
```

## 6. Autoencoders para anomalías

Un autoencoder es una [[Neural Networks]] que aprende a reconstruir sus entradas a través de un cuello de botella (bottleneck). La premisa: el modelo aprende a reconstruir bien los patrones "normales" (porque son la mayoría en entrenamiento) pero falla al reconstruir anomalías. El **reconstruction error** (MSE entre input y output) es el score de anomalía.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Construir autoencoder
input_dim = X_train.shape[1]
encoding_dim = 3

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Entrenar solo con datos normales
X_train_normal = X_train[y_train == 0]

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    verbose=0
)

# Calcular reconstruction error
reconstructions = autoencoder.predict(X_test, verbose=0)
mse = np.mean((X_test - reconstructions) ** 2, axis=1)

# Elegir threshold: percentil 99 del error en train normal
train_reconstructions = autoencoder.predict(X_train_normal, verbose=0)
train_mse = np.mean((X_train_normal - train_reconstructions) ** 2, axis=1)
threshold = np.percentile(train_mse, 99)

y_pred_ae = mse > threshold

precision_ae, recall_ae, _ = precision_recall_curve(y_test, mse)
pr_auc_ae = auc(recall_ae, precision_ae)

print(f"Threshold (P99 train): {threshold:.4f}")
print(f"Autoencoder - Anomalías detectadas: {y_pred_ae.sum()}/{y_test.sum()}")
print(f"Autoencoder - PR-AUC: {pr_auc_ae:.4f}")
```

Salida esperada:
```
Threshold (P99 train): 0.8921
Autoencoder - Anomalías detectadas: 12/15
Autoencoder - PR-AUC: 0.8560
```

**Autoencoder vs Isolation Forest:**
- Autoencoder captura relaciones no lineales complejas.
- Autoencoder requiere más datos y tuning (arquitectura, learning rate, epochs).
- Isolation Forest es deterministico y no requiere GPU.
- Autoencoder puede ajustar el threshold dinámicamente con el reconstruction error.

## 7. Evaluación

En detección de anomalías con desbalance extremo, la curva ROC es engañosa: el área bajo ROC puede ser alta incluso con muchos falsos positivos porque los negativos (normales) son muchísimos. Usa curvas Precision-Recall (PR).

```python
def evaluate_anomaly_detector(y_true, y_scores, model_name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # Top-k precision (asumiendo que sabemos que hay ~15 anomalías)
    k = y_true.sum()
    top_k_idx = np.argsort(y_scores)[-k:]
    precision_at_k = y_true[top_k_idx].mean()

    print(f"{model_name}:")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    print(f"  Precision@{k}: {precision_at_k:.4f}")

evaluate_anomaly_detector(y_test, -scores, "Isolation Forest")
evaluate_anomaly_detector(y_test, -lof_scores, "LOF")
evaluate_anomaly_detector(y_test, y_scores_hbos, "HBOS")
evaluate_anomaly_detector(y_test, mse, "Autoencoder")
```

Salida esperada:
```
Isolation Forest:
  PR-AUC:      0.8721
  Precision@15: 0.8667
LOF:
  PR-AUC:      0.8385
  Precision@15: 0.8000
HBOS:
  PR-AUC:      0.7931
  Precision@15: 0.7333
Autoencoder:
  PR-AUC:      0.8560
  Precision@15: 0.8000
```

**Métricas clave:**
- **PR-AUC**: área bajo la curva precision-recall. La métrica estándar en anomalías.
- **Precision@k**: de los top-k con mayor score, cuántas son realmente anomalías. Útil cuando tienes un presupuesto fijo de revisiones humanas.
- **Recall@k**: de todas las anomalías, cuántas aparecen en top-k.

## 8. Common Mistakes

### Threshold arbitrario
Usar `contamination=0.1` sin saber la proporción real de anomalías puede producir miles de falsos positivos (si la real es 0.001%) o no detectar nada (si la real es 5%). Ajusta el threshold con validación cruzada o con conocimiento del negocio.

### Contaminación en entrenamiento
Si entrenas un Isolation Forest o autoencoder con datos que contienen anomalías no etiquetadas, el modelo aprende que las anomalías son "normales" y las reconstruye bien. Siempre limpia el conjunto de entrenamiento o usa métodos robustos a contaminación (como `contamination` parameter ajustable).

### No separar train/val/test
En anomalías es tentador usar todos los datos para entrenar porque las etiquetas son escasas. Pero sin un conjunto de validación independiente, no puedes saber si tu threshold o tus hiperparámetros funcionan. Usa al menos un split temporal (time-based) para simular producción.

### Evaluar con ROC en lugar de PR
Como se mencionó arriba, ROC es engañoso con desbalance extremo. Un AUC-ROC de 0.99 puede esconder una precision de 0.01 si el recall es bajo y los negativos son masivos.

### Ignorar contexto temporal
En fraude bancario, las transacciones no son i.i.d. Un patrón normal hoy puede ser anómalo mañana ([[Data & Concept Drift]]). Los modelos de anomalías estáticos requieren retraining periódico.

## Resumen

| Método | Tipo | Escalabilidad | Dimensiones | Interpretabilidad |
|--------|:----:|:-------------:|:-----------:|:-----------------:|
| Isolation Forest | Basado en árboles | Alta (O(n log n)) | Alta | Media |
| LOF | Basado en densidad | Baja (O(n²)) | Baja-media | Alta |
| HBOS | Basado en histogramas | Muy alta (O(n×d)) | Alta | Muy alta |
| Autoencoder | Basado en redes neuronales | Media (GPU) | Muy alta | Baja |

En producción, la mayoría de sistemas usan Isolation Forest como primera línea por su balance velocidad-precisión, y Autoencoders para detectar anomalías sutiles y no lineales. La evaluación con PR-AUC y Precision@k es esencial para tomar decisiones de negocio informadas.

## Check Your Understanding

1. ¿Por qué Isolation Forest puede aislar anomalías con menos splits que puntos normales?
2. ¿Qué tipo de anomalía detecta LOF que Isolation Forest podría perder?
3. ¿Cuál es la principal limitación de HBOS?
4. ¿Por qué un autoencoder entrenado solo con datos normales falla al reconstruir anomalías?
5. ¿Por qué es preferible PR-AUC sobre ROC-AUC en detección de anomalías?
6. ¿Qué problema causa tener anomalías no etiquetadas en el conjunto de entrenamiento?
7. ¿Por qué las anomalías contextuales son difíciles de detectar con métodos estáticos?

<!--
1. Porque las anomalías son pocas y diferentes. En un árbol de aislamiento, cada split aleatorio separa el espacio. Las anomalías, al estar aisladas, requieren menos particiones para quedar solas en una hoja.
2. Anomalías locales: puntos que son normales en una región densa global pero anómalos respecto a su vecindario específico. LOF mide densidad local mientras que Isolation Forest mira el aislamiento global.
3. Asume independencia entre features. Si hay correlaciones fuertes (ej. "edad" y "años de experiencia"), HBOS no las captura y puede marcar combinaciones normales como anómalas.
4. El autoencoder aprende el manifold de los datos normales. Al tener un cuello de botella (bottleneck) pequeño, solo puede representar las variaciones principales. Las anomalías caen fuera de ese manifold y su reconstrucción es pobre, resultando en un reconstruction error alto.
5. ROC usa falsos positivos (FP) en el denominador, que es enorme en clases desbalanceadas. Un modelo puede tener muchos FP pero el ratio FP/N sigue siendo bajo dando un ROC engañoso. PR usa precisión = TP/(TP+FP), que penaliza directamente los FP.
6. Contaminación: el modelo aprende que las anomalías son "normales" y las reconstruye/clasifica bien. En autoencoders, el reconstruction error de anomalías será bajo y no las detectarás.
7. Los métodos estáticos no modelan el contexto (hora, ubicación, estacionalidad). Una anomalía contextual como "gasto alto a las 3 AM" requiere comparar dentro del mismo contexto temporal, no contra la distribución global.
-->

## Where to Go Next

- [[Unsupervised Learning]] — Fundamentos del aprendizaje no supervisado
- [[Supervised Learning]] — Alternativa cuando hay suficientes anomalías etiquetadas
- [[Model Evaluation]] — Métricas robustas para evaluar clasificadores
- [[Feature Engineering]] — Creación de features para mejorar detección
- [[Time Series Anomaly Detection]] — Métodos específicos para series temporales
- [[Neural Networks]] — Base de autoencoders y arquitecturas deep para anomalías
- [[Data & Concept Drift]] — Monitoreo de modelos en producción
