---
tags:
  - causal-inference
  - machine-learning
  - advanced
  - treatment-effects
status: seedling
created: 2026-06-28
---

# Causal ML

## 1. Escenario de aprendizaje

Lanzas una campaña de descuentos del 20% para un segmento de clientes. Comparas las ventas del grupo tratado contra el grupo control y ves un incremento del 15%. Reportas el éxito. Tu CMO pregunta: "¿Cuánto de ese incremento es realmente causado por el descuento y no porque los clientes tratados ya compraban más?"

No tienes respuesta. El problema: sesgo de selección. Los clientes que recibieron el descuento no fueron asignados aleatoriamente — fueron elegidos porque ya eran los más activos. Un simple t-test no puede separar correlación de causalidad.

Necesitas [[Causal Inference]] con ML para estimar **heterogeneous treatment effects (HTE)**: no solo el efecto promedio, sino cómo varía el impacto en cada segmento. Esta nota cubre la práctica de estimar efectos causales con modelos de [[Supervised Learning]].

---

## 2. ¿Por qué no basta con regresión?

### 2.1 Confounders no observados

Un confounder afecta tanto al tratamiento como al resultado. Si no lo incluyes en el modelo, el coeficiente del tratamiento está sesgado.

```
Confounder (U)
   ↙        ↘
Tratamiento → Resultado (Y)
```

Ejemplo: clientes más leales (no observado) son más propensos a recibir descuentos y a comprar más. Sin medir lealtad, el efecto estimado del descuento está inflado.

### 2.2 Selection bias

El tratamiento no se asigna aleatoriamente. Los grupos tratado y control difieren en características que afectan el resultado. La regresión ajusta por observables, pero cualquier diferencia en no observables sesga la estimación.

### 2.3 Heterogeneous effects

La regresión lineal estima un **Average Treatment Effect (ATE)**. Pero el descuento puede funcionar muy bien en jóvenes y no funcionar en mayores. Necesitamos estimar efectos condicionales: CATE.

---

## 3. Conditional Average Treatment Effect (CATE)

### 3.1 Definición formal

```
CATE(x) = E[Y(1) - Y(0) | X = x]
```

Donde:
- `Y(1)`: resultado potencial bajo tratamiento
- `Y(0)`: resultado potencial bajo control
- `X = x`: perfil del individuo

### 3.2 El problema fundamental de la inferencia causal

Para cada individuo, observamos solo un resultado potencial (el que ocurrió). El otro es **contrafactual**. No podemos calcular CATE directamente — necesitamos estimadores.

```python
import numpy as np
import pandas as pd
from numpy.random import randn, uniform, binomial

# Simular datos donde el efecto es heterogéneo
np.random.seed(42)
n = 1000

X = pd.DataFrame({
    'age': randn(n),
    'income': randn(n),
    'engagement': randn(n)
})

# Tratamiento depende de X (selection bias)
propensity = 1 / (1 + np.exp(-(0.5 * X['age'] + 0.3 * X['engagement'] - 0.2)))
T = binomial(1, propensity)

# Outcome: efecto de tratamiento varía con edad
Y0 = 0.5 * X['age'] + 0.3 * X['income'] + randn(n) * 0.1
Y1 = Y0 + 2.0 + 1.5 * (X['age'] > 0)  # mayor efecto en age > 0
Y = Y0 + T * (Y1 - Y0)

# ¿Qué estima una regresión lineal simple?
import statsmodels.api as sm
X_reg = sm.add_constant(pd.DataFrame({'T': T, 'age': X['age'], 'income': X['income']}))
model = sm.OLS(Y, X_reg).fit()
print(model.params)
```

**Salida esperada:**

```
const        -0.02
T             2.45
age           0.52
income        0.31
```

El coeficiente de T (2.45) sobreestima el ATE verdadero (~2.0) porque T está correlacionado con age (confounding). Una regresión ajusta linealmente, pero no modela la heterogeneidad del efecto.

---

## 4. Meta-Learners

Los meta-learners usan modelos de ML para estimar CATE sin modificar el algoritmo base.

### 4.1 S-Learner (Single Model)

Entrena un solo modelo con T como feature adicional.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

# S-Learner
X_with_T = X.copy()
X_with_T['T'] = T

model_s = GradientBoostingRegressor(n_estimators=100, max_depth=3)
model_s.fit(X_with_T, Y)

# Predecir CATE: diferencia entre Y(1) y Y(0) para cada x
X_1 = X.copy()
X_1['T'] = 1
X_0 = X.copy()
X_0['T'] = 0

cate_s = model_s.predict(X_1) - model_s.predict(X_0)
print(f"S-Learner CATE medio: {cate_s.mean():.3f}")
```

### 4.2 T-Learner (Two Models)

Entrena dos modelos separados: uno para el grupo tratado, otro para el grupo control.

```python
# T-Learner
model_t1 = GradientBoostingRegressor(n_estimators=100, max_depth=3)
model_t0 = GradientBoostingRegressor(n_estimators=100, max_depth=3)

model_t1.fit(X[T == 1], Y[T == 1])
model_t0.fit(X[T == 0], Y[T == 0])

cate_t = model_t1.predict(X) - model_t0.predict(X)
print(f"T-Learner CATE medio: {cate_t.mean():.3f}")
```

### 4.3 X-Learner

Combina S-Learner y T-Learner con un paso de imputación de efectos:

1. Entrena S-Learner para propensity score
2. Entrena T-Learner (modelos separados)
3. Imputa efectos individuales: `D(1) = Y(1) - Y_hat(0)`, `D(0) = Y_hat(1) - Y(0)`
4. Entrena modelos de efectos para tratados y controles
5. Combina ponderando por propensity score

Implementación con la librería `causalml`:

```python
from causalml.inference.meta import XLearner

xl = XLearner(
    learner=GradientBoostingRegressor(n_estimators=100, max_depth=3),
    control_learner=GradientBoostingRegressor(n_estimators=100, max_depth=3),
    treatment_learner=GradientBoostingRegressor(n_estimators=100, max_depth=3)
)

cate_x = xl.fit_predict(X, T, Y)
print(f"X-Learner CATE medio: {cate_x.mean():.3f}")
```

**Salida esperada:** Entre los 3 meta-learners, X-Learner suele tener menor sesgo cuando la asignación al tratamiento es desbalanceada o la relación entre X e Y difiere entre grupos.

---

## 5. Causal Forests

Causal Forests adaptan [[Ensemble Methods]] (Random Forest) para estimar CATE.

### 5.1 Honest Splitting

A diferencia de un Random Forest tradicional, Causal Forest divide los datos en dos submuestras para cada árbol:

- **Splitting sample:** decide los cortes óptimos maximizando la heterogeneidad del efecto
- **Estimation sample:** estima el CATE en cada hoja

Esto evita que los mismos datos que deciden la estructura del árbol también estimen el efecto, reduciendo sesgo.

```python
from econml.grf import CausalForest

cf = CausalForest(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

cf.fit(X, T, Y)
cate_cf = cf.effect(X)
print(f"Causal Forest CATE medio: {cate_cf.mean():.3f}")

# Importancia de variables para heterogeneidad
print("Feature importance:", cf.feature_importances_)
```

### 5.2 Interpretación

Causal Forest no solo estima CATE, sino que también identifica qué variables explican la heterogeneidad del efecto (feature importance). Útil para segmentación automática: ¿en qué grupo funciona el tratamiento?

---

## 6. Double ML

Double ML (Chernozhukov et al., 2018) usa ML para eliminar el sesgo de confounders mediante **ortogonalización**.

### 6.1 Idea central

1. Estimar `Y ~ X` → residual `Y_tilde = Y - Y_hat`
2. Estimar `T ~ X` → residual `T_tilde = T - T_hat`
3. Regresionar `Y_tilde ~ T_tilde` → estimación insesgada del ATE

Los residuales eliminan la variación explicada por `X`, usando [[Regularization]] implícita para evitar overfitting.

### 6.2 Implementación con `econml`

```python
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingRegressor(n_estimators=100),
    model_final=LassoCV(cv=5),
    discrete_treatment=True,
    cv=5
)

dml.fit(Y, T, X=X)
ate = dml.effect(X)
cate_dml = dml.effect(X)

print(f"Double ML ATE: {dml.ate_:.3f}")
print(f"Intervalo de confianza: {dml.ate_interval_:.3f}")
```

**Salida esperada:**

```
Double ML ATE: 2.12
Intervalo de confianza: (1.89, 2.35)
```

Double ML produce intervalos de confianza válidos asintóticamente y es robusto a errores de especificación en los modelos de ML.

### 6.3 Ventajas sobre meta-learners

- No asume forma funcional del efecto del tratamiento
- Intervalos de confianza válidos
- Doble robustez: basta con que uno de los dos modelos (Y o T) esté bien especificado

---

## 7. DoWhy

DoWhy (Microsoft Research) es un framework causal que unifica modelado de supuestos, estimación y refutación.

### 7.1 Cuatro pasos

```python
import dowhy
from dowhy import CausalModel
import pandas as pd

# 1. Modelar supuestos causales con un DAG
data = pd.DataFrame({
    'Y': Y, 'T': T,
    'age': X['age'], 'income': X['income'], 'engagement': X['engagement']
})

model = CausalModel(
    data=data,
    treatment='T',
    outcome='Y',
    graph="""
    digraph {
        age -> T;
        age -> Y;
        income -> Y;
        engagement -> T;
        engagement -> Y;
        T -> Y;
    }
    """
)

# 2. Identificar el estimando
identified_estimand = model.identify_effect()
print(identified_estimand)

# 3. Estimar el efecto causal
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
print(f"Estimación: {estimate.value:.3f}")

# 4. Refutar con Placebo Test
refutation = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute"
)
print(f"Placebo test p-value: {refutation.new_effect}")
```

**Salida esperada:**

```
Estimación: 2.45
   (Causal estimand: estimación del efecto de T en Y)
Placebo test p-value: 0.02
   (Placebo Treatment Refuter: El efecto estimado cambió significativamente)
```

### 7.2 Tests de refutación

DoWhy incluye múltiples tests para evaluar la robustez:

- **Placebo Treatment:** reasignar aleatoriamente el tratamiento; el efecto debe ser cero
- **Random Common Cause:** agregar un confounder aleatorio; el efecto no debe cambiar
- **Data Subset Validation:** estimar en subconjuntos; el efecto debe ser consistente
- **Unobserved Confounders:** simular sensibilidad a confounders no observados

### 7.3 Integración con `econml`

```python
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    method_params={
        'model_y': GradientBoostingRegressor(n_estimators=100),
        'model_t': GradientBoostingRegressor(n_estimators=100),
        'cv': 5
    }
)
print(f"Estimación DML via DoWhy: {estimate.value:.3f}")
```

---

## 8. Common Mistakes

### Ignorar overlap
- **Error:** Estimar CATE en regiones del espacio X donde todos o nadie reciben tratamiento
- **Solución:** Verificar que `0 < P(T=1|X) < 1` para todo X. Si hay violaciones graves, restringir el análisis al soporte común

### Condicionar en colliders
- **Error:** Incluir una variable que es causada tanto por T como por Y (collider), introduciendo sesgo de selección
- **Solución:** Usar DAGs para identificar colliders. No condicionar en variables que son descendientes comunes de T e Y

### No evaluar supuestos de identificación
- **Error:** Reportar CATE sin verificar si los supuestos (no confounders, no interference, consistencia) se cumplen
- **Solución:** Siempre ejecutar tests de refutación (DoWhy) y análisis de sensibilidad

### Usar ML sin control por overfitting en el CATE
- **Error:** Meta-learners con modelos muy complejos (GBM profundo) pueden sobreajustar el CATE y producir estimaciones ruidosas
- **Solución:** Validación cruzada en el CATE, podar árboles, usar [[Regularization]] en modelos base

### Confundir ATE con CATE
- **Error:** Reportar el ATE como si aplicara a todos los individuos
- **Solución:** Reportar distribución del CATE (percentiles, segmentos), no solo el promedio

---

## Resumen

| Método | Tipo | Ventaja clave | Riesgo |
|---|---|---|---|
| Regresión | Paramétrico | Simple, interpretable | Sesgo por confounders no lineales |
| S-Learner | Meta-learner | Un solo modelo | Ignora T si es débil |
| T-Learner | Meta-learner | Modelos especializados | Ineficiente si grupos pequeños |
| X-Learner | Meta-learner | Balance efecto/imputación | Complejidad |
| Causal Forest | Árbol | Heterogeneidad automática | Requiere N grande |
| Double ML | Ortogonalización | Intervalos válidos, robusto | Complejidad computacional |
| DoWhy | Framework | Supuestos explícitos + refutación | Requiere DAG correcto |

La inferencia causal con ML no es magia — es disciplina: explicitas supuestos, estimas con métodos robustos, y refutas sistemáticamente.

---

## Check Your Understanding

**1. ¿Qué diferencia a CATE de ATE?**
<!-- ATE es el efecto promedio en toda la población. CATE es el efecto condicional a un perfil X = x. CATE permite segmentación. -->

**2. ¿Por qué Double ML usa residuales en lugar de las variables originales?**
<!-- Los residuales eliminan la variación explicada por confounders, permitiendo que la estimación final del efecto sea ortogonal a X y por tanto insesgada. -->

**3. ¿Qué es "honest splitting" en Causal Forests?**
<!-- Usar submuestras diferentes para decidir los cortes y para estimar los efectos en cada hoja. Reduce el sesgo de usar los mismos datos para estructura y estimación. -->

**4. ¿Cuál es la diferencia entre S-Learner y T-Learner?**
<!-- S-Learner incluye T como feature en un solo modelo. T-Learner entrena modelos separados para cada grupo. T-Learner funciona mejor si T tiene interacciones fuertes con X. -->

**5. ¿Qué verifica un placebo test en DoWhy?**
<!-- Reasigna aleatoriamente el tratamiento (rompe la relación causal). Si el efecto estimado sigue siendo significativo, hay sesgo. -->

---

## Where to Go Next

- [[Causal Inference]] — Fundamentos teóricos, DAGs, supuestos de identificación
- [[Experimental Design]] — Cómo diseñar RCTs para evitar sesgo de selección
- [[Ensemble Methods]] — Base de los Causal Forests
- [[Supervised Learning]] — Modelos base usados en meta-learners
- [[A-B Testing]] — Comparación entre experimentos y estudios observacionales
- [[Statistics]] — Inferencia estadística, intervalos de confianza, tests de hipótesis
- [[Data Product Thinking]] — Cómo comunicar resultados causales a stakeholders
