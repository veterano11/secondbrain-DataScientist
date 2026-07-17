---
tags:
  - statistics
  - causal-inference
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

No puedes hacer un experimento controlado (A/B test) — es muy caro o poco ético. Pero necesitas saber si X causa Y. La inferencia causal ofrece métodos para estimar causalidad con datos observacionales.

---

## 1. Correlación ≠ causalidad

Dos variables pueden correlacionarse sin que una cause la otra por **confounders** (variables ocultas que afectan ambas). El **Simpson's paradox** ocurre cuando una tendencia desaparece o se invierte al estratificar por un confounder.

```python
import pandas as pd
import numpy as np

# Simpson's paradox simulado
df = pd.DataFrame({
    'tratamiento': np.random.choice([0, 1], 200),
    'genero': np.random.choice(['M', 'F'], 200),
    'recuperacion': np.random.rand(200)
})
print(df.groupby('tratamiento')['recuperacion'].mean())
```

---

## 2. DAGs (Directed Acyclic Graphs)

Los DAGs modelan visualmente los supuestos causales. El **backdoor criterion** identifica qué variables debemos condicionar para estimar el efecto causal: bloquear todos los caminos "backdoor" (confounders) sin abrir colliders.

```
# Estructura conceptual:
# Z → X → Y   (mediador — no condicionar)
# X ← Z → Y   (confounder — condicionar)
# X → Z ← Y   (collider — no condicionar)
```

---

## 3. Matching

El **propensity score matching** estima la probabilidad de recibir tratamiento dadas las covariables P(T=1|X) y empareja unidades tratadas con controles similares en ese score.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.random.rand(100, 3)
T = np.random.binomial(1, 0.5, 100)
modelo = LogisticRegression()
modelo.fit(X, T)
propensity_scores = modelo.predict_proba(X)[:, 1]
print(propensity_scores[:5])
```

**Salida esperada:**
```
[0.52 0.48 0.55 0.51 0.49]
```

**Nearest neighbor** empareja al vecino más cercano. **Caliper matching** impone una distancia máxima aceptable.

---

## 4. Diff-in-Diff

Compara el cambio pre-post entre grupo tratamiento y control. El supuesto clave son **tendencias paralelas**: sin tratamiento, ambos grupos habrían evolucionado de la misma forma.

```python
import pandas as pd
import statsmodels.formula.api as smf

# Datos simulados: grupo (tratamiento=1 / control=0), tiempo (post=1 / pre=0)
df = pd.DataFrame({
    'Y': [10, 12, 15, 18, 9, 11, 12, 13],
    'grupo': [1, 1, 1, 1, 0, 0, 0, 0],
    'post': [0, 1, 0, 1, 0, 1, 0, 1]
})
modelo = smf.ols('Y ~ grupo * post', data=df).fit()
print(modelo.params['grupo:post'])
```

**Salida esperada:**
```
1.0
```

La interacción `grupo:post` es el efecto causal estimado.

---

## 5. Instrumental Variables (IV)

Un **instrumento (Z)** afecta al tratamiento (X) pero no tiene efecto directo en el outcome (Y) salvo a través de X. Es útil cuando hay confounders no observables.

```python
from linearmodels.iv import IV2SLS
import pandas as pd
import numpy as np

n = 1000
Z = np.random.normal(0, 1, n)
X = 0.5 * Z + np.random.normal(0, 1, n)
Y = 2.0 * X + np.random.normal(0, 1, n)
df = pd.DataFrame({'Y': Y, 'X': X, 'Z': Z})
modelo = IV2SLS.from_formula('Y ~ 1 + [X ~ Z]', df)
resultados = modelo.fit()
print(resultados.params['X'])
```

**Salida esperada:**
```
2.01
```

---

## 6. Common Mistakes

1. **Ignorar confounders**: omitir variables que afectan tratamiento y outcome.
2. **Condicionar en colliders**: variable causada por dos o más factores; condicionar en ella crea sesgo de selección.
3. **DAGs incompletos**: un modelo causal incorrecto produce estimaciones sesgadas.

```python
# Collider bias conceptual:
# capacidad -> aceptación <- suerte
# Condicionar en "aceptación" correlaciona capacidad y suerte espuriamente
```

---

## Resumen

1. Correlación no implica causalidad — los confounders pueden explicar relaciones espurias.
2. Los DAGs ayudan a visualizar supuestos causales y guían qué condicionar.
3. Matching (propensity score) crea grupos comparables a partir de datos observacionales.
4. Diff-in-Diff compara cambios pre-post entre grupos bajo supuesto de tendencias paralelas.
5. Variables instrumentales estiman causalidad cuando hay confounders no observables.
6. Condicionar en colliders o ignorar confounders produce estimaciones sesgadas.

---

## Check Your Understanding

1. ¿Qué es un confounder? <!-- Variable que afecta tanto al tratamiento como al outcome -->
2. ¿Qué criterio determina qué variables condicionar en un DAG? <!-- Backdoor criterion -->
3. ¿Cuál es el supuesto clave de Diff-in-Diff? <!-- Tendencias paralelas entre grupos antes del tratamiento -->
4. ¿Qué requisitos debe cumplir un instrumento válido? <!-- Relevancia (afecta al tratamiento) y exogeneidad (no afecta directamente al outcome) -->

---

## Where to Go Next

- [[Experimental Design]]
- [[A-B Testing]]
- [[Bayesian Inference]]
- [[Statistics]]
