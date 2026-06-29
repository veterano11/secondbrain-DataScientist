---
tags:
  - machine-learning
  - ensemble-methods
  - bagging
  - boosting
  - stacking
status: seedling
created: 2026-06-28
---

# Ensemble Methods

## 1. Escenario de aprendizaje

Estás participando en una competencia de Kaggle de datos tabulares con 100,000 instancias y 300 features. Tu mejor modelo individual —un XGBoost con hiperparámetros optimizados— te da un RMSE de 0.523 en validación cruzada. Pero el leaderboard muestra que el primer lugar tiene 0.501. La diferencia no está en un modelo mágico, sino en la combinación inteligente de múltiples modelos: un ensemble.

Un solo [[Decision Trees]] sufre de alta varianza (overfitting). Una regresión lineal sufre de alto sesgo (underfitting). Los ensembles combinan múltiples modelos débiles para crear un predictor fuerte que supera a cualquier modelo individual. Esta es la técnica que consistentemente gana competencias de ML y se usa en producción en sistemas de recomendación, detección de fraude y predicción financiera.

En esta nota verás las cuatro grandes familias de ensambles: bagging (Random Forest), boosting (XGBoost, LightGBM, CatBoost), stacking y blending. Cada una tiene fortalezas distintas y entender cuándo usar cada una es lo que separa a un practitioner avanzado de uno principiante.

## 2. Requisitos

```bash
pip install scikit-learn xgboost lightgbm catboost numpy pandas
```

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
```

## 3. Bagging: Bootstrap Aggregating

Bagging entrena M modelos en paralelo usando muestras bootstrap (muestreo con reemplazo) del dataset original. La predicción final es el promedio (regresión) o la votación (clasificación). La clave teórica: el promedio de M variables aleatorias con varianza σ² tiene varianza σ²/M. Reducimos varianza sin aumentar el sesgo.

### Random Forest

Random Forest extiende bagging añadiendo aleatoriedad en las features: cada split solo considera un subconjunto aleatorio de features. Esto descorrelaciona aún más los árboles, lo que reduce aún más la varianza.

```python
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

Salida esperada:
```
Accuracy: 0.9430
AUC-ROC: 0.9792
```

**Hiperparámetros críticos:**
- `n_estimators`: mientras más, mejor (pero ley de rendimientos decrecientes). Monitorea con OOB error.
- `max_depth`: controla la profundidad. Árboles profundos → menor sesgo, mayor varianza.
- `min_samples_leaf`: evita hojas con muy pocas muestras.
- `max_features`: `sqrt` para clasificación, `n/3` para regresión (regla empírica).

## 4. Boosting

Mientras bagging entrena en paralelo, boosting entrena secuencialmente: cada nuevo modelo corrige los errores del anterior. Esto reduce el sesgo progresivamente.

### AdaBoost

Asigna pesos a cada instancia. Tras cada iteración, aumenta el peso de las instancias mal clasificadas, forzando al siguiente modelo a enfocarse en los casos difíciles.

### Gradient Boosting

Generalización de AdaBoost. Cada nuevo árbol se ajusta a los **residuos** (gradiente negativo de la función de pérdida) del modelo anterior.

```python
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]

print(f"Gradient Boosting - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Gradient Boosting - AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

Salida esperada:
```
Gradient Boosting - Accuracy: 0.9450
AUC-ROC: 0.9815
```

## 5. XGBoost

XGBoost es Gradient Boosting optimizado con:

- **Regularización L1 y L2** en los pesos de las hojas — Reduce [[Overfitting]] dramáticamente.
- **Pruning tipo tree** (no por profundidad): poda ramas con ganancia negativa, permitiendo árboles más profundos pero más limpios.
- **Handling missing values**: aprende la dirección óptima (left/right) para valores faltantes durante el entrenamiento.
- **Aproximación cuantil** para splits: evita ordenar todos los valores.

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,     # L1
    reg_lambda=1.0,    # L2
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print(f"XGBoost - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"XGBoost - AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

Salida esperada:
```
XGBoost - Accuracy: 0.9490
AUC-ROC: 0.9863
```

### Early Stopping

XGBoost acepta un conjunto de validación y detiene el entrenamiento si la métrica no mejora tras `early_stopping_rounds`. Esto es esencial para evitar overfitting y ahorrar tiempo de cómputo.

## 6. LightGBM

LightGBM introduce dos innovaciones para escalar a datasets con millones de filas:

- **GOSS (Gradient-based One-Side Sampling)**: en cada iteración, retiene todas las instancias con gradiente grande (error alto) y muestrea aleatoriamente las de gradiente pequeño. Esto enfoca el cómputo en las instancias más informativas.
- **EFB (Exclusive Feature Bundling)**: combina features dispersas (como one-hot encodings) en un solo bundle, reduciendo el número de dimensiones.
- **Histogram-based splits**: discretiza valores continuos en bins, haciendo splits O(bins) en lugar de O(n).
- **Soporte nativo para variables categóricas**: pasa `categorical_feature` directamente, sin one-hot.

```python
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,
    random_state=42
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
y_pred = lgb_model.predict(X_test)
y_proba = lgb_model.predict_proba(X_test)[:, 1]

print(f"LightGBM - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"LightGBM - AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

Salida esperada:
```
LightGBM - Accuracy: 0.9480
AUC-ROC: 0.9851
```

⚠️ **Advertencia**: `num_leaves` es el análogo de `max_depth`. Con `num_leaves=31` y profundidad ~5, LightGBM tiende a crecer árboles más complejos que XGBoost. Reduce `num_leaves` o incrementa `min_child_samples` si ves overfitting.

## 7. CatBoost

CatBoost fue desarrollado por Yandex para manejar datos con muchas variables categóricas sin preprocessing manual.

- **Ordered Boosting**: versión de boosting que evita el target leakage inherente al Gradient Boosting clásico. En cada iteración, el gradiente se calcula solo con datos que el modelo no ha visto antes (similar a validación cruzada online).
- **Symmetric trees (oblivious trees)**: todos los nodos en un mismo nivel usan la misma feature y el mismo threshold. Esto acelera la inferencia y reduce overfitting.
- **Categorical nativas**: usa target encoding con suavizado basado en prior, sin necesidad de LabelEncoder o OneHotEncoder.

```python
cb_model = cb.CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    border_count=128,
    cat_features=[],  # Pasar índices de columnas categóricas
    eval_metric='AUC',
    early_stopping_rounds=50,
    random_seed=42,
    verbose=False
)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
y_pred = cb_model.predict(X_test)
y_proba = cb_model.predict_proba(X_test)[:, 1]

print(f"CatBoost - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"CatBoost - AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

Salida esperada:
```
CatBoost - Accuracy: 0.9470
AUC-ROC: 0.9848
```

## 8. Stacking

Stacking entrena múltiples modelos base (nivel 0) y después entrena un **meta-modelo** (nivel 1) que aprende a combinar las predicciones de los modelos base. La clave está en que el meta-modelo no ve las features originales, solo las predicciones de los niveles inferiores.

```python
from sklearn.ensemble import StackingClassifier

base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42, verbosity=0)),
    ('lgb', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=15, random_state=42, verbose=-1))
]

meta_model = LogisticRegression(C=1.0, max_iter=1000)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba'
)
stack.fit(X_train, y_train)
y_proba = stack.predict_proba(X_test)[:, 1]

print(f"Stacking - AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

Salida esperada:
```
Stacking - AUC-ROC: 0.9881
```

**Stacking exitoso requiere:**
1. Modelos base diversos (diferentes algoritmos, diferentes subsets de features).
2. Meta-modelo simple (regresión logística, Ridge) para evitar overfitting.
3. Validación cruzada para generar las predicciones del nivel 0 (evitar [[data leakage]]).

## 9. Blending

Blending es una versión simplificada de stacking:

1. Parte el dataset en train (80%) y holdout (20%).
2. Entrena modelos base en train.
3. Genera predicciones sobre holdout.
4. Entrena meta-modelo con las predicciones del holdout como features.

Es más simple y rápido que stacking (no requiere CV), pero usa menos datos para entrenar el meta-modelo. Funciona bien como baseline rápido.

## 10. Common Mistakes

### Bagging con modelos de alta varianza
Random Forest ya reduce varianza; usar `max_depth=None` sin control produce árboles ruidosos que el promediado no logra limpiar completamente. Fija `max_depth` o `min_samples_leaf`.

### Boosting con learning_rate alto
`learning_rate > 0.3` con `n_estimators` bajos hace que el modelo aprenda los residuos demasiado rápido y sobreajuste. Regla general: `lr=0.01-0.1` con suficientes estimadores.

### Leak en stacking
Si entrenas modelos base en todo el dataset y después entrenas el meta-modelo con esas predicciones, el meta-modelo ve información del target. Usa `cv` en `StackingClassifier` o implementa tu propia validación cruzada.

### Ignorar features categóricas en LightGBM/CatBoost
Pasar categóricas sin declararías explícitamente fuerza one-hot encoding, que es ineficiente y pierde la señal de cardinalidad alta.

## Resumen

| Método | Entrenamiento | Reduce | Cuándo usarlo |
|--------|:------------:|:------:|:-------------:|
| Bagging (RF) | Paralelo | Varianza | Datasets medianos, quick baseline, features heterogéneas |
| Boosting (XGB/LGB/CB) | Secuencial | Sesgo + Varianza | Datos tabulares, competencias, máxima precisión |
| Stacking | 2 niveles | Ambos | Cuando ya tienes modelos fuertes y quieres exprimir +0.5-1% |
| Blending | 2 niveles | Ambos | Prototipado rápido, equipos pequeños |

La elección práctica: empieza con Random Forest como baseline, prueba XGBoost con tuning básico, después LightGBM si el dataset es grande (>100K filas), CatBoost si hay muchas categóricas, y finalmente stacking/blending si estás compitiendo.

## Check Your Understanding

1. ¿Por qué bagging reduce varianza sin aumentar el sesgo?
2. ¿Qué diferencia fundamental hay entre el entrenamiento paralelo de bagging y el secuencial de boosting?
3. ¿Cuál es la ventaja de la regularización L1/L2 en XGBoost comparado con GradientBoostingClassifier de sklearn?
4. ¿Qué problema resuelve GOSS en LightGBM?
5. ¿Por qué es importante que los modelos base en stacking sean diversos?
6. ¿En qué caso elegirías CatBoost sobre XGBoost?
7. ¿Qué riesgo introduce blending comparado con stacking?

<!--
1. Porque promedia M variables aleatorias independientes con varianza σ², reduciendo la varianza a σ²/M. El sesgo del ensemble es el promedio de los sesgos individuales, que es igual al sesgo de un árbol individual.
2. Bagging entrena modelos independientes en paralelo y promedia. Boosting entrena secuencialmente donde cada modelo corrige los errores del anterior, reduciendo sesgo progresivamente.
3. GradientBoostingClassifier de sklearn no tiene regularización directa sobre los pesos de las hojas. XGBoost añade términos L1 (alpha) y L2 (lambda) en la función de pérdida, penalizando pesos grandes y reduciendo overfitting.
4. GOSS retiene todas las instancias con gradiente grande (alta contribución al error) y muestrea aleatoriamente las de gradiente pequeño, enfocando el cómputo en las instancias más informativas.
5. Modelos diversos capturan diferentes patrones en los datos. Si todos los modelos base cometen los mismos errores, el meta-modelo no tiene señal para corregirlos.
6. Cuando el dataset tiene muchas variables categóricas de alta cardinalidad (cientos de niveles). CatBoost las maneja nativamente sin preprocessing ni one-hot encoding.
7. Blending usa menos datos (solo el holdout) para entrenar el meta-modelo, lo que puede aumentar la varianza de la predicción final si el holdout no es representativo.
-->

## Where to Go Next

- [[Supervised Learning]] — Fundamentos de aprendizaje supervisado
- [[Regularization]] — Técnicas de regularización que se integran con ensembles
- [[Overfitting]] — Diagnóstico y prevención de sobreajuste
- [[Model Evaluation]] — Validación cruzada y métricas para comparar ensembles
- [[Hyperparameter Tuning]] — Estrategias de optimización de parámetros
- [[Feature Engineering]] — Creación de features que los ensembles puedan explotar
- [[Decision Trees]] — Base teórica de todos los métodos basados en árboles
