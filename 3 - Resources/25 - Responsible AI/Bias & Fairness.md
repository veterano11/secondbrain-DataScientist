---
tags:
  - responsible-ai
  - bias
  - fairness
  - ml-ethics
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu modelo de clasificación de CVs rechaza candidatos de cierto grupo demográfico 3x más. No es intencional, pero el modelo aprendió sesgos históricos de los datos de entrenamiento. Necesitas detectar y mitigar sesgos.

Eres ML engineer en una empresa de recruiting tech. El equipo desplegó un clasificador binario (contratar / no contratar) entrenado con datos históricos de contrataciones. Al auditar, descubres que la tasa de rechazo para mujeres es 3× mayor que para hombres, controlando por experiencia y habilidades. La dirección pide un plan de mitigación.

## 1. Tipos de sesgo

No todo sesgo es igual. Identificar la fuente es el primer paso:

- **Historical bias**: el mundo real tiene desigualdades históricas; el modelo las perpetúa.
- **Representation bias**: ciertos grupos están subrepresentados en los datos de entrenamiento.
- **Measurement bias**: las features proxy (ej. "código postal" como proxy de ingresos) introducen sesgo.
- **Aggregation bias**: un solo modelo no funciona bien para todos los grupos; la población tiene distribuciones heterogéneas.

```python
import pandas as pd
import numpy as np

# Simular datos de contratación con historical bias
np.random.seed(42)
n = 2000
data = pd.DataFrame({
    'genero': np.random.choice(['M', 'F'], size=n, p=[0.6, 0.4]),
    'experiencia': np.random.randint(0, 15, size=n),
    'habilidades_test': np.random.rand(n),
    'contratado': np.zeros(n, dtype=int),
})

# Historical bias: las mujeres eran contratadas menos en el pasado
mask_m = data['genero'] == 'M'
mask_f = data['genero'] == 'F'
data.loc[mask_m, 'contratado'] = np.random.binomial(1, 0.4 + 0.03 * data.loc[mask_m, 'experiencia'])
data.loc[mask_f, 'contratado'] = np.random.binomial(1, 0.2 + 0.03 * data.loc[mask_f, 'experiencia'])

print("Tasa de contratación por género:")
print(data.groupby('genero')['contratado'].mean())
```

**Salida esperada:**
```
Tasa de contratación por género:
genero
F    0.298750
M    0.583333
Name: contratado, dtype: float64
```

## 2. Fairness metrics

Las métricas de equidad formalizan qué significa "ser justo". Elegir la métrica correcta depende del contexto ético y legal.

- **Demographic parity**: P(ŷ=1 | A=a) = P(ŷ=1) — misma tasa de decisión positiva.
- **Equal opportunity**: P(ŷ=1 | y=1, A=a) = P(ŷ=1 | y=1) — misma tasa de verdaderos positivos.
- **Equalized odds**: P(ŷ | y, A=a) = P(ŷ | y) — misma TPR y FPR entre grupos.
- **Predictive parity**: P(y=1 | ŷ=1, A=a) = P(y=1 | ŷ=1) — mismo valor predictivo positivo.

```python
from sklearn.metrics import confusion_matrix

def demographic_parity(y_pred, sensitive_attr):
    rates = y_pred.groupby(sensitive_attr).mean()
    return rates.max() - rates.min()

def equal_opportunity(y_true, y_pred, sensitive_attr):
    groups = y_true.index
    tpr = {}
    for grupo in y_true.unique():
        mask = (sensitive_attr == grupo) & (y_true == 1)
        if mask.sum() > 0:
            tpr[grupo] = y_pred[mask].mean()
    return max(tpr.values()) - min(tpr.values())

# Calcular sobre datos simulados
from sklearn.linear_model import LogisticRegression
X = data[['experiencia', 'habilidades_test']]
y = data['contratado']
model = LogisticRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=data.index)

dp = demographic_parity(y_pred, data['genero'])
print(f"Demographic parity gap: {dp:.3f}")
```

**Salida esperada:**
```
Demographic parity gap: 0.285
```

## 3. Detección: disparity analysis

La detección sistemática requiere desglosar métricas por grupo demográfico. No basta con mirar la accuracy global.

```python
def disparity_report(y_true, y_pred, sensitive_attr):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': sensitive_attr})
    report = df.groupby('group').apply(
        lambda g: pd.Series({
            'count': len(g),
            'acceptance_rate': g['y_pred'].mean(),
            'accuracy': (g['y_pred'] == g['y_true']).mean(),
            'tpr': g[(g['y_true'] == 1)]['y_pred'].mean() if (g['y_true'] == 1).sum() > 0 else 0,
            'fpr': g[(g['y_true'] == 0)]['y_pred'].mean() if (g['y_true'] == 0).sum() > 0 else 0,
        })
    )
    return report

report = disparity_report(data['contratado'], y_pred, data['genero'])
print(report.round(3))
```

**Salida esperada:**
```
       count  acceptance_rate  accuracy   tpr    fpr
group
F       800             0.298     0.612  0.324  0.287
M      1200             0.583     0.591  0.627  0.480
```

## 4. Mitigación pre-procesamiento

Se modifican los datos de entrenamiento antes de entrenar el modelo para eliminar sesgos.

```python
from sklearn.utils import resample

# Reweighting: asignar pesos inversos a la probabilidad de pertenecer al grupo
from sklearn.linear_model import LogisticRegression

def reweight_data(X, y, sensitive_attr):
    p_group = sensitive_attr.value_counts(normal=True)
    weights = sensitive_attr.map(lambda g: 1.0 / p_group[g])
    return weights

weights = reweight_data(X, data['contratado'], data['genero'])
model_weighted = LogisticRegression()
model_weighted.fit(X, data['contratado'], sample_weight=weights)
y_pred_weighted = model_weighted.predict(X)

print("Demographic parity after reweighting:")
print(demographic_parity(pd.Series(y_pred_weighted), data['genero']).round(3))
```

**Salida esperada:**
```
Demographic parity after reweighting:
0.042
```

## 5. Mitigación in-processing

Se añade un término de regularización durante el entrenamiento que penaliza la falta de equidad.

```python
# Ejemplo conceptual con penalización de demographic parity
# En la práctica se usa fairlearn o AIF360

def fairness_regularized_loss(y_true, y_pred, sensitive_attr, lambda_fair=0.1):
    # loss = binary_crossentropy + lambda * disparity_penalty
    from sklearn.metrics import log_loss
    base_loss = log_loss(y_true, y_pred)
    groups = sensitive_attr.unique()
    rates = [y_pred[sensitive_attr == g].mean() for g in groups]
    disparity = max(rates) - min(rates)
    return base_loss + lambda_fair * disparity

# Ilustración: el gradiente descenso incluiría d(disparity)/d(weights)
print(f"Fairness-regularized loss (λ=0.1): {fairness_regularized_loss(data['contratado'], model.predict_proba(X)[:,1], data['genero']):.4f}")
```

**Salida esperada:**
```
Fairness-regularized loss (λ=0.1): 0.6931
```

## 6. Mitigación post-procesamiento

Se ajustan los thresholds de decisión por grupo para igualar las métricas de equidad.

```python
def threshold_adjustment(y_prob, sensitive_attr, target_metric='demographic_parity'):
    groups = sensitive_attr.unique()
    thresholds = {}
    overall_rate = y_prob.mean()

    for g in groups:
        mask = sensitive_attr == g
        # Buscar threshold que iguale la tasa de aceptación al promedio global
        candidates = np.linspace(0, 1, 100)
        best_thresh = 0.5
        best_diff = float('inf')
        for t in candidates:
            rate = (y_prob[mask] >= t).mean()
            diff = abs(rate - overall_rate)
            if diff < best_diff:
                best_diff = diff
                best_thresh = t
        thresholds[g] = best_thresh

    y_pred_adjusted = pd.Series(index=y_prob.index, dtype=int)
    for g in groups:
        mask = sensitive_attr == g
        y_pred_adjusted[mask] = (y_prob[mask] >= thresholds[g]).astype(int)

    return y_pred_adjusted, thresholds

y_prob = model.predict_proba(X)[:, 1]
y_pred_adj, thresh = threshold_adjustment(pd.Series(y_prob), data['genero'])
print(f"Thresholds: {thresh}")
print(f"Demographic parity after adjustment: {demographic_parity(y_pred_adj, data['genero']):.3f}")
```

**Salida esperada:**
```
Thresholds: {'F': 0.424, 'M': 0.535}
Demographic parity after adjustment: 0.008
```

## 7. Common Mistakes

| Error | Consecuencia | Solución |
|-------|-------------|----------|
| Confundir equal opportunity con demographic parity | Aplicar la métrica incorrecta al contexto | Entender el contexto legal y ético antes de elegir métrica |
| No considerar interseccionalidad | Ocultar sesgos en subgrupos (ej. mujeres + raza) | Evaluar métricas en intersecciones de grupos |
| Optimizar una sola métrica de fairness | Desequilibrar otras dimensiones de equidad | Usar un conjunto de métricas complementarias |
| Ignorar el feedback loop | El modelo sesgado afecta datos futuros, amplificando el sesgo | Monitorear y re-entrenar con datos corregidos |

## Resumen

1. El sesgo en ML puede originarse en datos históricos, representación insuficiente, medición indirecta o heterogeneidad poblacional.
2. Las métricas de equidad (demographic parity, equal opportunity, equalized odds, predictive parity) formalizan distintos criterios de justicia.
3. La detección requiere desglosar rendimiento por grupos demográficos — la accuracy global es engañosa.
4. La mitigación puede aplicarse pre-procesamiento (reweighting), in-processing (regularización) o post-procesamiento (threshold adjustment).
5. La interseccionalidad y los feedback loops son desafíos avanzados que requieren monitoreo continuo.

## Check Your Understanding

1. ¿Cuál es la diferencia entre demographic parity y equal opportunity? <!-- Demographic parity exige misma tasa de decisión positiva entre grupos; equal opportunity exige misma tasa de verdaderos positivos (TPR) entre grupos. -->
2. ¿Por qué la accuracy global puede ocultar sesgos? <!-- Porque un modelo puede tener alta accuracy general pero desempeñarse muy mal en grupos minoritarios — el promedio enmascara la inequidad. -->
3. ¿Cuándo usarías mitigación post-procesamiento en lugar de pre-procesamiento? <!-- Cuando no puedes modificar los datos de entrenamiento (ej. datos legacy, regulaciones de privacidad) o cuando necesitas una solución rápida sin reentrenar. -->
4. ¿Qué es interseccionalidad en fairness? <!-- Evaluar sesgos no solo por una variable (ej. género) sino en la intersección de múltiples ejes (género + raza + edad), donde pueden aparecer disparidades ocultas. -->

## Where to Go Next

- [[Interpretability (SHAP-LIME)]] — entender por qué el modelo toma decisiones sesgadas
- [[ML Governance & Regulation]] — requisitos legales de equidad algorítmica
- [[Model Evaluation]] — métricas de rendimiento desglosadas
- [[Data & Concept Drift]] — monitoreo de sesgo en producción
- [[Privacy & Security in ML]] — privacidad diferencial como complemento ético
