---
tags: [bayesian, probabilistic-programming, pymc]
status: growing
created: 2026-06-27
---

# Probabilistic Programming

## 1. Escenario de aprendizaje

Querés modelar la relación entre edad e ingreso, pero sabés que la varianza del ingreso aumenta con la edad (heteroscedasticidad). Un [[Supervised Learning|modelo lineal clásico]] ignora esto. Con PyMC podés especificar exactamente el modelo que querés: media lineal, varianza que crece, y obtener la incertidumbre completa de los parámetros.

Probabilistic Programming frameworks (PyMC, Stan) automatizan la inferencia bayesiana. Escribís el modelo, ellos se encargan del sampling.

## 2. El flujo de trabajo PyMC

```text
1. Definir modelo (priors + likelihood)
2. Samplear (MCMC / NUTS)
3. Diagnosticar (R-hat, ESS, trace plot)
4. Analizar (summary, posterior predictive checks)
```

## 3. Ejemplo 1: Regresión lineal bayesiana

```python
import pymc as pm
import numpy as np

# Datos sintéticos
np.random.seed(42)
X = np.random.normal(0, 1, 100)
y = 2 + 1.5 * X + np.random.normal(0, 0.5, 100)

with pm.Model() as modelo_lineal:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfCauchy("sigma", beta=2)

    # Likelihood
    mu = alpha + beta * X
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Samplear
    trace = pm.sample(1000, tune=1000, chains=4, random_seed=42)

# Resultados
print(pm.summary(trace, var_names=["alpha", "beta", "sigma"]))
```

**Salida esperada:**
```
         mean    sd   hdi_3%  hdi_97%
alpha   2.01   0.06   1.90    2.12
beta    1.49   0.05   1.40    1.59
sigma   0.52   0.04   0.45    0.59
```

Los valores verdaderos ($\alpha=2$, $\beta=1.5$, $\sigma=0.5$) están dentro del HDI. El sampling recuperó los parámetros correctamente.

## 4. Posterior Predictive Checks

¿El modelo genera datos parecidos a los observados?

```python
with modelo_lineal:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

y_pred = ppc["y_obs"]  # shape: (4000, 100)

# Comparar distribución de datos reales vs predichos
print(f"Real: media={y.mean():.2f}, std={y.std():.2f}")
print(f"Pred: media={y_pred.mean():.2f}, std={y_pred.std():.2f}")
```

## 5. Ejemplo 2: Modelo jerárquico (multi-level)

Escenario: medimos el efecto de un fármaco en 10 hospitales, cada uno con ~50 pacientes. Queremos un efecto general que "tira" los efectos individuales hacia la media (partial pooling).

```python
with pm.Model() as modelo_jerarquico:
    # Hiperpriors
    mu_efecto = pm.Normal("mu_efecto", mu=0, sigma=10)
    tau = pm.HalfCauchy("tau", beta=2)

    # Efecto por hospital (shrinkage hacia la media)
    efecto_h = pm.Normal("efecto_h", mu=mu_efecto, sigma=tau, shape=n_hospitales)

    # Likelihood
    sigma = pm.HalfCauchy("sigma", beta=2)
    mu = efecto_h[hospital_idx]  # cada paciente tiene el efecto de su hospital
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(1000, tune=1000)
```

**Beneficio**: hospitales con pocos pacientes tienen estimaciones más estables (shrinked hacia la media global). Hospitales con muchos pacientes tienen estimaciones cercanas a su media empírica.

## 6. Ejemplo 3: Bayesian A/B Testing

```python
with pm.Model() as ab_test:
    p_A = pm.Beta("p_A", alpha=1, beta=1)
    p_B = pm.Beta("p_B", alpha=1, beta=1)

    obs_A = pm.Binomial("obs_A", n=n_A, p=p_A, observed=conv_A)
    obs_B = pm.Binomial("obs_B", n=n_B, p=p_B, observed=conv_B)

    delta = pm.Deterministic("delta", p_B - p_A)

    trace = pm.sample(2000)

# Probabilidad de que B sea mejor
prob = (trace["posterior"]["delta"] > 0).mean()
print(f"P(B > A) = {prob:.3f}")
```

## 7. Diagnósticos

```python
import arviz as az

data = az.from_pymc(trace)

# R-hat y ESS
print(az.summary(data)[["r_hat", "ess_bulk"]])

# Trace plots
az.plot_trace(data, var_names=["alpha", "beta"])

# Posterior distribution
az.plot_posterior(data, var_names=["beta"], hdi_prob=0.94)
```

## 8. PyMC vs Stan vs NumPyro

| Framework | Ventaja | Desventaja |
|-----------|---------|------------|
| **PyMC** | API Pythonica, excelentes diagnósticos | Más lento que NumPyro |
| **Stan** | Muy rápido, gran comunidad | Sintaxis propia, menos Pythonic |
| **NumPyro** | GPU, JAX, muy rápido | Menos maduro, menos ejemplos |

## 9. Common Mistakes

1. **No suficiente warmup**: NUTS necesita calentar. Min 1000 tuning steps. Menos → biased samples.
2. **R-hat mal interpretado**: < 1.01 para todos los parámetros. Si alpha tiene 1.02 y beta 1.001, la cadena de alpha no convergió.
3. **Priors demasiado amplios**: Normal(0, 1000) parece "no informativo" pero asigna alta probabilidad a valores irreales. Usá priors weakly informative.
4. **No hacer PPC**: el modelo puede samplear bien pero generar datos irreales. PPC detecta misspecification.

## 10. Check Your Understanding

1. En el modelo jerárquico, un hospital con 5 pacientes tiene efecto_h estimado cerca de mu_efecto. ¿Por qué? (Partial pooling — pocos datos → la información del grupo domina)
2. ¿Qué ventaja tiene usar PyMC sobre escribir tu propio Metropolis-Hastings? (NUTS automático, diagnóstico integrado, GPU)
3. ¿Cuándo usarías NumPyro en vez de PyMC? (Cuando necesitás GPU o modelos muy grandes)

## 11. Summary

[[Statistics|Probabilistic Programming]] permite especificar modelos bayesianos complejos sin derivar matemáticas. PyMC es el estándar Python: defines priors + likelihood, sampleas con NUTS, diagnosticas con ArviZ. Los modelos jerárquicos son el caso de uso estrella (partial pooling). Siempre verificar convergencia (R-hat, ESS) y hacer posterior predictive checks.

## 12. Where to Go Next

- [[Markov Chain Monte Carlo]] — cómo funciona el sampler por dentro
- [[Bayesian Inference]] — teoría de priors, likelihood, posterior
- [[A-B Testing]] — aplicar modelos bayesianos a experiments
- [[Gaussian Processes]] — modelos no paramétricos con PyMC
