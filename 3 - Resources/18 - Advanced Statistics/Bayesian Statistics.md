---
tags:
  - statistics
  - bayesian
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Lanzas un producto nuevo. Cada día ves más datos. Quieres actualizar tu creencia sobre la tasa de conversión en tiempo real. La estadística bayesiana te permite hacer esto de forma natural: prior → likelihood → posterior.

---

## 1. Teorema de Bayes

$$P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}$$

- **Prior** P(θ): creencia inicial sobre el parámetro.
- **Likelihood** P(D|θ): probabilidad de los datos dado el parámetro.
- **Posterior** P(θ|D): creencia actualizada después de ver los datos.
- **Evidence** P(D): probabilidad marginal de los datos (normalización).

```python
# Conceptual: prior Beta(2, 10), likelihood Binomial
# Posterior es Beta(2 + exitos, 10 + fracasos)
```

---

## 2. Conjugate priors

Un prior es **conjugado** si la posterior pertenece a la misma familia que el prior. Permite actualización analítica sin MCMC.

| Likelihood | Prior conjugado | Posterior |
|---|---|---|
| Binomial | Beta(α, β) | Beta(α + k, β + n − k) |
| Normal (σ² conocida) | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) |
| Poisson | Gamma(α, β) | Gamma(α + Σx, β + n) |

```python
from scipy.stats import beta

# Prior: Beta(2, 10) — creencia pesimista (~16% de conversión)
prior_a, prior_b = 2, 10

# Datos: 30 conversiones de 200 visitas
exitos, visitas = 30, 200

# Posterior: Beta(2 + 30, 10 + 170) = Beta(32, 180)
post_a = prior_a + exitos
post_b = prior_b + (visitas - exitos)

media_post = post_a / (post_a + post_b)
print(f"Media posterior: {media_post:.3f}")

# IC credible 95%
ic_inf = beta.ppf(0.025, post_a, post_b)
ic_sup = beta.ppf(0.975, post_a, post_b)
print(f"IC 95%: [{ic_inf:.3f}, {ic_sup:.3f}]")
```

**Salida esperada:**
```
Media posterior: 0.151
IC 95%: [0.105, 0.205]
```

---

## 3. Bayesian updating

La gran ventaja bayesiana: la posterior de hoy es el prior de mañana. Esto permite **actualización secuencial** en tiempo real.

```python
from scipy.stats import beta

# Día 1: prior Beta(2, 10), datos: 3 exitos / 20 visitas
a, b = 2, 10
a += 3; b += 17  # posterior día 1

# Día 2: la posterior del día 1 es el prior del día 2
# datos: 5 exitos / 30 visitas
a += 5; b += 25  # posterior día 2

print(f"Día 2 — media: {a/(a+b):.3f}, IC 95%: ({beta.ppf(0.025, a, b):.3f}, {beta.ppf(0.975, a, b):.3f})")
```

**Salida esperada:**
```
Día 2 — media: 0.156, IC 95%: (0.107, 0.212)
```

---

## 4. Credible intervals vs Confidence intervals

| | Credible Interval (Bayes) | Confidence Interval (Frecuentista) |
|---|---|---|
| Interpretación | "95% de probabilidad de que θ esté en este intervalo" | "95% de los intervalos así construidos contendrán θ" |
| Lo que varía | θ es aleatorio | θ es fijo, el intervalo es aleatorio |
| Requiere prior | Sí | No |

```python
# El credible interval responde P(θ ∈ [a, b] | datos) = 0.95
# El confidence interval responde P(IC contiene θ) = 0.95
```

---

## 5. Bayes Factors

El **Bayes Factor** (BF) cuantifica la evidencia a favor de un modelo frente a otro:

$$BF_{10} = \frac{P(D \mid H_1)}{P(D \mid H_0)}$$

| BF₁₀ | Evidencia contra H₀ |
|---|---|
| 1–3 | Débil |
| 3–10 | Moderada |
| 10–30 | Fuerte |
| > 30 | Muy fuerte |

```python
# Ejemplo conceptual: comparar si la conversión es > 0.10
# H₀: θ = 0.10, H₁: θ ~ Beta(1, 1)
# Con 30 exitos en 200 visitas, BF₁₀ ≈ 3.2 → evidencia moderada
```

---

## 6. Priors

- **Informativos**: basados en estudios previos (ej: Beta(50, 950) para conversión ~5%).
- **Weakly informative**: algo de información pero permitiendo que los datos dominen (ej: Beta(2, 10)).
- **Flat/uniforme**: "no informativo" — la posterior es proporcional a la likelihood (ej: Beta(1, 1)).

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

thetas = np.linspace(0, 1, 100)
plt.plot(thetas, beta.pdf(thetas, 50, 950), label='Informativo')
plt.plot(thetas, beta.pdf(thetas, 2, 10), label='Weakly informative')
plt.plot(thetas, beta.pdf(thetas, 1, 1), label='Flat')
plt.legend()
plt.show()
```

**Salida esperada:** Tres curvas de densidad Beta con distinta concentración.

---

## 7. Common Mistakes

1. **Priors demasiado informativos**: un prior muy fuerte puede dominar incluso con muchos datos.
2. **Confundir credible interval con confidence interval**: son filosóficamente diferentes.
3. **Usar un prior flat sin pensar**: no siempre es "no informativo" — puede afectar la posterior en escalas transformadas.

```python
# Prior flat Beta(1,1) parece neutral, pero
# en escala log-odds no es uniforme
# Siempre examina el impacto del prior en tus conclusiones
```

---

## Resumen

1. El teorema de Bayes actualiza creencias combinando prior y likelihood.
2. Los conjugate priors permiten actualización analítica y secuencial.
3. Los credible intervals tienen una interpretación probabilística directa.
4. Los Bayes Factors cuantifican evidencia entre modelos.
5. La elección del prior importa — weakly informative es un buen punto de partida.
6. No confundas credible intervals (bayesianos) con confidence intervals (frecuentistas).

---

## Check Your Understanding

1. ¿Qué significa que un prior sea "conjugado"? <!-- La posterior pertenece a la misma familia que el prior -->
2. ¿Cuál es la diferencia clave entre un credible interval y un confidence interval? <!-- El credible interval expresa probabilidad sobre θ; el confidence interval sobre el procedimiento -->
3. ¿Qué indica un Bayes Factor de 15? <!-- Evidencia fuerte a favor de H₁ sobre H₀ -->
4. ¿Por qué un prior flat no siempre es "no informativo"? <!-- Puede ser informativo bajo transformaciones no lineales del parámetro -->

---

## Where to Go Next

- [[Bayesian Inference]]
- [[Markov Chain Monte Carlo]]
- [[Probabilistic Programming]]
- [[A-B Testing]]
- [[Experimental Design]]
