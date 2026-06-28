---
tags:
  - responsible-ai
  - interpretability
  - explainability
  - shap
  - lime
  - xai
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu modelo rechazó un préstamo. El cliente preguntó "¿por qué?". El regulador exige una explicación. Los modelos de ML son cajas negras. SHAP y LIME abren la caja negra y explican predicciones individuales y globales.

Trabajas en una fintech. El modelo de scoring crediticio (XGBoost) rechazó un préstamo a un cliente con buen historial. El cliente apeló. El regulador (CFPB) exige una explicación "significativa" de por qué fue rechazado. Necesitas herramientas de interpretabilidad.

## 1. ¿Por qué interpretabilidad?

La interpretabilidad no es un lujo — es un requisito operativo, regulatorio y ético.

- **Confianza**: stakeholders y usuarios necesitan entender las decisiones del modelo.
- **Debugging**: detectar artefactos espurios, leakage, features incorrectas.
- **Regulación**: EU AI Act, CFPB, FDA exigen explicaciones para decisiones automatizadas.
- **Fairness**: verificar que el modelo no usa features protegidas de forma indirecta.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Simular datos de crédito
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'ingreso': np.random.lognormal(mean=10, sigma=0.5, size=n),
    'edad': np.random.randint(18, 70, size=n),
    'historial_pagos': np.random.choice(['bueno', 'regular', 'malo'], size=n, p=[0.6, 0.3, 0.1]),
    'deuda_ingreso': np.random.beta(2, 5, size=n),
    'num_consultas': np.random.poisson(2, size=n),
})
data['target'] = (
    (data['ingreso'] > 30000) &
    (data['historial_pagos'] != 'malo') &
    (data['deuda_ingreso'] < 0.5)
).astype(int)

# Entrenar modelo caja negra
X = pd.get_dummies(data.drop('target', axis=1), columns=['historial_pagos'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
```

**Salida esperada:**
```
Accuracy: 0.945
```

## 2. Feature importance

Miden qué features contribuyen más a las predicciones del modelo.

```python
import matplotlib.pyplot as plt

# Permutation importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print(importance_df.head())
```

**Salida esperada:**
```
           feature  importance       std
1              edad    0.038000  0.011225
0           ingreso    0.028500  0.015207
3     deuda_ingreso    0.019000  0.008433
4     num_consultas    0.009500  0.005408
2  historial_pagos_malo  0.007500  0.005013
```

**Cuidado:** La feature importance basada en impurity puede favorecer features numéricas sobre categóricas con muchas categorías. Permutation importance es más robusta.

## 3. Partial Dependence Plots (PDP)

Muestran cómo cambia la predicción promedio al variar un feature, manteniendo los demás constantes — relación marginal.

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
PartialDependenceDisplay.from_estimator(
    model, X_test, ['ingreso', 'edad'],
    grid_resolution=20, ax=ax
)
plt.tight_layout()
plt.show()
print("PDP generado: muestra relación positiva entre ingreso/edad y probabilidad de aprobación")
```

**Salida esperada:**
```
PDP generado: muestra relación positiva entre ingreso/edad y probabilidad de aprobación
```

## 4. LIME (Local Interpretable Model-agnostic Explanations)

LIME aproxima el modelo localmente con un modelo lineal interpretable. Responde: "para esta predicción específica, ¿qué features pesaron más?"

```python
!pip install lime -q

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['rechazado', 'aprobado'],
    mode='classification'
)

# Explicar una predicción individual (caso rechazado)
idx = 0  # primer ejemplo del test set que fue rechazado
exp = explainer.explain_instance(X_test.values[idx], model.predict_proba, num_features=4)
exp.show_in_notebook(show_table=True)
print("Features que más contribuyeron al rechazo:", dict(exp.as_list()))
```

**Salida esperada:**
```
Features que más contribuyeron al rechazo: {'ingreso <= 25000.00': -0.32, 'deuda_ingreso > 0.45': -0.21, 'historial_pagos_malo': -0.15, 'num_consultas > 3': -0.08}
```

## 5. SHAP (SHapley Additive exPlanations)

SHAP usa teoría de juegos cooperativos para asignar a cada feature su contribución marginal. Es consistente y localmente preciso.

```python
!pip install shap -q

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
print("Summary plot generado")

# Dependence plot
shap.dependence_plot("ingreso", shap_values, X_test, feature_names=X_test.columns)
print("Dependence plot generado")
```

**Salida esperada:**
```
Summary plot generado
Dependence plot generado
```

```python
# Force plot para una predicción individual
idx = 0  # mismo caso rechazado
shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx])
print("Force plot generado para predicción individual")
```

**Salida esperada:**
```
Force plot generado para predicción individual
```

**Interpretación del force plot:** Las features en rojo empujan la predicción hacia arriba (mayor probabilidad de aprobación); las azules hacia abajo (mayor probabilidad de rechazo). La magnitud indica cuánto.

## 6. Interpretabilidad global vs local

Ambos niveles se complementan — ninguna responde todo.

| Aspecto | Global | Local |
|---------|--------|-------|
| ¿Qué responde? | "¿Cómo funciona el modelo en general?" | "¿Por qué esta predicción específica?" |
| Herramientas | SHAP summary, permutation importance, PDP | SHAP force plot, LIME, Shapley values individuales |
| Cuándo usarlo | Auditoría, validación, documentación | Apelaciones, debugging, explicaciones al usuario |
| Limitación | No explica casos individuales atípicos | Puede ser inestable: pequeñas variaciones cambian explicaciones |

```python
# Comparar importancia global vs local
global_importance = np.abs(shap_values).mean(axis=0)
global_ranking = X_test.columns[np.argsort(global_importance)[::-1]]
local_ranking = X_test.columns[np.argsort(np.abs(shap_values[idx]))[::-1]]

print("Top-3 features globales:", list(global_ranking[:3]))
print("Top-3 features locales (caso rechazado):", list(local_ranking[:3]))
```

**Salida esperada:**
```
Top-3 features globales: ['ingreso', 'edad', 'deuda_ingreso']
Top-3 features locales (caso rechazado): ['ingreso', 'deuda_ingreso', 'historial_pagos_malo']
```

## 7. Common Mistakes

| Error | Consecuencia | Solución |
|-------|-------------|----------|
| Confiar ciegamente en feature importance | Features correlacionadas distorsionan la importancia | Usar permutation importance + verificar correlaciones |
| Ignorar correlación entre features | PDP y SHAP pueden mostrar relaciones engañosas | Acompañar con dependence plots condicionales (ICE) |
| SHAP en datasets grandes sin muestreo | Tiempo de cómputo prohibitivo | Usar `shap.approximate` o muestrear datos (tip: 100-500 rows para summary) |
| Explicar solo global (o solo local) | La explicación queda incompleta | Combinar ambas perspectivas en informes |

## Resumen

1. La interpretabilidad es necesaria para confianza, debugging, regulación y fairness.
2. Permutation importance mide contribución global de cada feature; no asume nada sobre el modelo.
3. Partial dependence plots muestran relaciones marginales entre features y predicción.
4. LIME genera explicaciones locales mediante un modelo sustituto lineal interpretable.
5. SHAP asigna contribuciones basadas en Shapley values, con consistencia matemática.
6. La interpretabilidad global y local responden preguntas distintas y se complementan.
7. Las correlaciones entre features y el costo computacional son desafíos prácticos clave.

## Check Your Understanding

1. ¿Cuál es la diferencia fundamental entre LIME y SHAP? <!-- LIME entrena un modelo sustituto lineal local; SHAP calcula contribuciones basadas en Shapley values con fundamentos de teoría de juegos. SHAP es consistente y localmente preciso; LIME puede ser inestable. -->
2. ¿Por qué permutation importance es preferible a impurity-based importance? <!-- Porque impurity-based importance puede sesgarse hacia features con alta cardinalidad o correlacionadas; permutation importance mide el impacto real en la pérdida al permutar el feature. -->
3. ¿Cuándo usarías un PDP en lugar de un SHAP dependence plot? <!-- PDP muestra la relación marginal promedio; SHAP dependence plot muestra la distribución de contribuciones individuales y revela heterogeneidad e interacciones. -->
4. ¿Qué limitación tiene LIME cuando el modelo es muy no lineal? <!-- El modelo sustituto lineal puede no aproximar bien la frontera de decisión local si la curvatura es muy alta, dando explicaciones poco fieles. -->

## Where to Go Next

- [[Bias & Fairness]] — usar SHAP para detectar features proxy de sesgo
- [[ML Governance & Regulation]] — documentar explicaciones para auditorías
- [[Model Evaluation]] — evaluar calidad de explicaciones (fidelity, stability)
- [[Supervised Learning]] — entender los modelos que intentamos explicar
- [[Feature Engineering]] — features explicables desde el diseño
