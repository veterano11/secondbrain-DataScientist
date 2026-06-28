---
tags: [bayesian, mcmc, sampling, inference]
status: growing
created: 2026-06-27
---

# Markov Chain Monte Carlo

## 1. Escenario de aprendizaje

Querés estimar la posterior de un [[Statistics|modelo jerárquico]] con 50 parámetros. No hay conjugados, no hay solución cerrada. Necesitás **muestrear** de la posterior. MCMC construye una cadena de Markov cuya distribución estacionaria es la posterior.

MCMC es el motor detrás de PyMC, Stan, y la mayoría de inferencia bayesiana moderna.

## 2. El problema

La posterior $P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}$ tiene un denominador $P(D)$ que es una integral multidimensional:

$$P(D) = \int P(D|\theta) P(\theta) d\theta$$

Para modelos con más de ~5 parámetros, esa integral no se puede calcular analíticamente ni por cuadratura numérica.

**MCMC**: no calcula $P(D)$. Solo necesita el **numerador** $P(D|\theta) P(\theta)$ (no-normalizado) para generar muestras.

## 3. Metropolis-Hastings

El algoritmo más simple de MCMC:

```python
def metropolis(log_posterior, init, steps=10000, proposal_std=0.1):
    theta = init
    samples = [theta]
    aceptados = 0

    for _ in range(steps):
        # 1. Proponer nuevo valor
        propuesta = np.random.normal(theta, proposal_std)

        # 2. Calcular ratio de aceptación
        log_ratio = log_posterior(propuesta) - log_posterior(theta)
        ratio = np.exp(log_ratio)

        # 3. Aceptar o rechazar
        if np.random.random() < min(1, ratio):
            theta = propuesta
            aceptados += 1

        samples.append(theta)

    print(f"Tasa de aceptación: {aceptados/steps:.2%}")
    return np.array(samples)
```

**Aceptación ideal**: ~23-44%. Si aceptás demasiado, la propuesta es muy chica y explorás lento. Si aceptás muy poco, la propuesta es muy grande y rebotás.

**Problema**: Metropolis-Hastings escala mal a dimensiones altas (más de ~10 parámetros).

## 4. HMC (Hamiltonian Monte Carlo)

Usa [[Linear Algebra|gradientes]] para proponer saltos largos con alta aceptación. Piensa en $\theta$ como la posición de una partícula con energía potencial $-\log P(\theta|D)$.

```text
1. Muestrear momento ρ ~ N(0, M)
2. Simular dinámica Hamiltoniana por L pasos con step ε
3. Aceptar/rechazar basado en cambio de energía total
```

**Ventaja**: propuestas a mucha distancia con alta aceptación (incluso en altas dimensiones).
**Desventaja**: necesita gradiente de $\log P(\theta|D)$. Requiere ajustar step size $\epsilon$ y path length $L$.

## 5. NUTS (No-U-Turn Sampler)

HMC necesita que ajustes $L$ (cuántos pasos de simulación). NUTS automatiza $L$: corre la simulación hasta que empieza a dar la vuelta (U-turn).

```text
NUTS adapta automáticamente:
- ε (step size) → target acceptance rate (~0.8)
- M (mass matrix) → de la warmup
- L (path length) → por U-turn criterion
```

**NUTS es el sampler default de PyMC y Stan**. En la práctica, rara vez necesitás otro.

## 6. Gibbs Sampling

Cuando las condicionales completas son conocidas, sampleás un parámetro a la vez condicionado en los demás:

$$P(\theta_1 | \theta_2, ..., \theta_k, D)$$
$$P(\theta_2 | \theta_1, \theta_3, ..., \theta_k, D)$$

Sin tuning, sin rechazo. Ideal para modelos con estructura condicional (mixture models, LDA).

## 7. Diagnóstico de convergencia

MCMC no te da muestras independientes de la posterior. Te da una cadena correlacionada. Necesitás verificar que convergió.

### 7.1 Trace plot

```python
import matplotlib.pyplot as plt

# Graficar las muestras en orden
plt.plot(trace["beta"][:, 0])
plt.title("Trace plot de β[0]")
plt.ylabel("β[0]")
plt.xlabel("Iteración")
```

**Qué buscar**: la cadena debe verse como un "gusano peludo" — sin tendencias, sin estancamientos.

### 7.2 R-hat ($\hat{R}$)

Ejecutá **4+ cadenas** desde puntos de inicio diferentes. $\hat{R}$ compara varianza entre cadenas con varianza dentro de cada cadena:

$$\hat{R} = \sqrt{\frac{\text{Varianza entre cadenas}}{\text{Varianza dentro de cadenas}}}$$

- $\hat{R} < 1.01$: convergió (regla moderna). 
- $\hat{R} > 1.05$: no convergió — necesitás más iteraciones.

### 7.3 ESS (Effective Sample Size)

Mide cuántas muestras **independientes** equivalentes tenés. Si ESS = 50 de 1000 muestras, la cadena es muy autocorrelacionada.

```python
import arviz as az

data = az.from_pymc(trace)
summary = az.summary(data)
print(summary[["r_hat", "ess_bulk", "ess_tail"]])
```

## 8. Warmup (Burn-in)

Las primeras muestras dependen de la inicialización y no representan la posterior. Se descartan (típicamente 50% de las muestras).

```python
with model:
    trace = pm.sample(2000, tune=1000, chains=4)
    # 2000 draws, 1000 warmup, 4 chains
    # Total: 4 × 2000 = 8000 muestras (post-warmup)
```

## 9. Common Mistakes

1. **R-hat < 1.1 NO es suficiente**: el estándar moderno es < 1.01. Un R-hat de 1.05 puede esconder problemas.
2. **Ignorar ESS**: 5000 muestras con ESS 30 es como tener 30 muestras independientes. Corré más iteraciones.
3. **No ejecutar múltiples cadenas**: una cadena puede parecer convergida desde una inicialización engañosa. 4 cadenas mínimo.
4. **Flat priors en varianzas**: priors planos en $\sigma$ (ej: Uniform(0, ∞)) llevan a posteriores impropias. MCMC no converge. Usá HalfCauchy o InverseGamma.

## 10. Check Your Understanding

1. Tenés 4 cadenas, todas con R-hat = 1.003. ¿Qué significa?
2. ESS es 500 de 10000 muestras. ¿Cuántas muestras independientes equivalentes tenés?
3. Tu trace plot muestra un plateau de 1000 iteraciones seguido de un salto. ¿Qué problema tiene la cadena?

**Respuestas rápidas:**
1. Las cadenas convergieron al mismo lugar. Podés confiar en las estimaciones.
2. 500. Necesitás más iteraciones para reducir la autocorrelación.
3. La cadena está atascada en un modo y saltó a otro. Posible multimodalidad. Usá SMC o más chains.

## 11. Summary

MCMC genera muestras de la posterior cuando no hay solución analítica. Metropolis-Hastings es simple pero no escala. HMC usa gradientes para alta dimensionalidad. NUTS automatiza HMC y es el estándar. Gibbs samplea condicionales cuando están disponibles. Los diagnósticos (R-hat < 1.01, ESS alto) son obligatorios para confiar en los resultados.

## 12. Where to Go Next

- [[Probabilistic Programming]] — usar MCMC en la práctica con PyMC
- [[Bayesian Inference]] — qué estamos muestreando
- [[Gaussian Processes]] — inferencia con GP
- [[RL Fundamentals]] — MDP y cadenas de Markov comparten fundamentos
