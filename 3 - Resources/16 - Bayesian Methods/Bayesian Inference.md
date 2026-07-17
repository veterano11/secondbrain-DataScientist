---
tags: [bayesian, inference, probability, statistics]
status: growing
created: 2026-06-27
---

# Bayesian Inference

## 1. Escenario de aprendizaje

Hiciste un test A/B: versión A (control) tuvo 50 conversiones de 1000 visitas (5%). Versión B (variante) tuvo 60 conversiones de 1000 visitas (6%).

La pregunta frecuentista: "¿$p < 0.05$?" La pregunta bayesiana: "¿Cuál es la probabilidad de que B sea mejor que A?"

Bayesian Inference responde la segunda pregunta directamente, y de paso cuantifica la incertidumbre.

## 2. El teorema de Bayes

$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

| Componente | Nombre | Significa |
|-----------|--------|-----------|
| $P(\theta)$ | Prior | Lo que creemos antes de ver datos |
| $P(D\|\theta)$ | Likelihood | Probabilidad de los datos dado $\theta$ |
| $P(\theta\|D)$ | Posterior | Lo que creemos después de ver datos |

## 3. Paso a paso: Test A/B bayesiano

### 3.1 Prior

Antes de ver datos, nuestra creencia sobre la tasa de conversión: podría ser cualquier valor entre 0 y 1. Usamos una distribución Beta(1,1) (uniforme = ningún valor favorecido).

$$P(\theta) = \text{Beta}(1, 1)$$

### 3.2 Likelihood

Dado $\theta$ (tasa de conversión real), la probabilidad de observar $k$ conversiones en $n$ visitas:

$$P(k | \theta) = \text{Binomial}(n, \theta)$$

### 3.3 Posterior

Magia de los **conjugados**: Beta × Binomial = Beta.

$$P(\theta | k, n) = \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$$

```python
import numpy as np
from scipy import stats

# Datos
n_A, k_A = 1000, 50
n_B, k_B = 1000, 60

# Posterior: Beta(prior_alpha + exitos, prior_beta + fracasos)
prior_alpha, prior_beta = 1, 1
post_A = stats.beta(prior_alpha + k_A, prior_beta + n_A - k_A)
post_B = stats.beta(prior_alpha + k_B, prior_beta + n_B - k_B)

# 4. Probabilidad de que B > A
muestras_A = post_A.rvs(100_000)
muestras_B = post_B.rvs(100_000)
prob_B_mejor = (muestras_B > muestras_A).mean()
print(f"P(B > A) = {prob_B_mejor:.3f}")

# 5. Intervalo de credibilidad del 95% para B
print(f"B: 95% credible interval = ({post_B.ppf(0.025):.3f}, {post_B.ppf(0.975):.3f})")
```

**Salida esperada:**
```
P(B > A) = 0.823
B: 95% credible interval = (0.046, 0.076)
```

**Interpretación**: hay 82.3% de probabilidad de que B sea mejor que A. El intervalo de credibilidad para B es [4.6%, 7.6%].

> **Pregunta**: ¿cuántos datos más necesitás para llegar a 95%? Con más visitas, las distribuciones se estrechan. Con 10,000 visitas por grupo y las mismas tasas, P(B>A) ≈ 0.98.

## 4. Priors

La elección del prior es el punto más criticado de la estadística bayesiana. ¿Qué prior usar?

| Prior | Uso | Ejemplo |
|-------|-----|---------|
| **Flat/Uniforme** | "No sé nada" | Beta(1,1), Normal(0, 1e6) |
| **Weakly informative** | "Sé algo pero poco" | Beta(2,2): creencia débil que la tasa ronda 0.5 |
| **Informative** | Datos previos confiables | Beta(50, 950): estudio anterior con 5% de conversión |
| **Skeptical** | "Dudo que haya efecto grande" | Normal(0, 0.1) para efectos |

**Siempre hacé sensitivity analysis**: probá con priors diferentes. Si la conclusión cambia, tu resultado depende del prior más que de los datos.

## 5. Credible Intervals vs Confidence Intervals

La diferencia más importante entre [[Statistics|estadística frecuentista]] y bayesiana:

| | Intervalo de confianza | Intervalo de credibilidad |
|--|----------------------|--------------------------|
| Significado | "Si repitiéramos el experimento muchas veces, 95% de los intervalos contendrían el verdadero valor" | "Hay 95% de probabilidad de que el parámetro esté en este intervalo" |
| Interpretación intuitiva | No | Sí |

El bayesiano da la respuesta que la gente naturalmente quiere.

## 6. Bayesian Decision Theory

No solo obtenemos $P(\theta|D)$ — podemos usarla para tomar [[Model Evaluation|decisiones óptimas]].

$$a^* = \arg\min_a \int L(\theta, a) P(\theta | D) d\theta$$

| Pérdida | Decisión óptima |
|---------|-----------------|
| $(a - \theta)^2$ (error cuadrático) | Media de la posterior |
| $\|a - \theta\|$ (error absoluto) | Mediana de la posterior |
| 0-1 (acertar/errar) | Moda de la posterior (MAP) |

## 7. Common Mistakes

1. **Interpretar el intervalo de credibilidad como si fuera de confianza**: decir "95% de probabilidad de que el parámetro esté en este rango" solo es válido en estadística bayesiana.
2. **Prior demasiado fuerte con pocos datos**: con n=10, un prior informativo domina la posterior. Mostrar el resultado con prior débil también.
3. **MAP como único resumen**: el modo de la posterior (MAP) da un punto pero pierde toda la información de incertidumbre. Siempre reportá el intervalo también.
4. **No verificar robustez**: cambiá el prior y si la conclusión cambia, no confíes en ella.

## 8. Check Your Understanding

1. Test A/A (misma versión): 500 visitas, 25 conv. cada uno. ¿Qué esperás que dé P(B > A)?
2. ¿Por qué un prior Beta(1,1) se considera "no informativo"?
3. ¿Cuándo usarías un prior informativo en vez de uno plano?

**Respuestas rápidas:**
1. ~0.5. Si no hay diferencia real, las posteriores son casi iguales.
2. Es equivalente a Uniform(0,1) — todas las tasas igualmente probables.
3. Cuando tenés datos de un estudio previo confiable (ej: misma campaña, año anterior).

## 9. Resumen

La Inferencia Bayesiana actualiza creencias (prior → posterior) usando el teorema de Bayes. Los priors conjugados (Beta-Binomial, Normal-Normal) permiten actualización cerrada. La posterior da una distribución completa de probabilidad del parámetro, no solo un punto. Los intervalos de credibilidad son intuitivos ("95% de probabilidad"). La regla principal: siempre probar sensibilidad al prior.

## 10. Dónde Ir Ahora

- [[Programación Probabilística]] — Inferencia bayesiana con PyMC (sin conjugados)
- [[Cadena de Markov Monte Carlo]] — muestreo de posteriores arbitrarias
- [[Procesos Gaussianos]] — Regresión bayesiana no paramétrica
- [[Pruebas A/B]] — aplicar pruebas A/B bayesianas en producción
- [[Probabilidad]] — distribución Beta y Binomial en detalle
