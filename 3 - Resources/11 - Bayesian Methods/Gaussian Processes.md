---
tags: [bayesian, gaussian-processes, nonparametric, regression]
status: growing
created: 2026-06-27
---

# Gaussian Processes

## 1. Escenario de aprendizaje

Querés modelar una función desconocida $f(x)$ a partir de 20 puntos observados con ruido. No sabés si la función es lineal, cuadrática, o sinusoidal. Una regresión lineal asumiría una forma fija. Un GP no: aprende la complejidad de los datos y te da incertidumbre en cada predicción.

GPs son la herramienta ideal para optimización bayesiana, modelado de funciones costo, y [[Supervised Learning|regresión]] con datos escasos.

## 2. ¿Qué es un GP?

Un GP es una distribución sobre funciones. Cualquier conjunto finito de puntos evaluados en un GP tiene una distribución conjunta Gaussiana.

$$\mu(x) = \mathbb{E}[f(x)]$$
$$k(x, x') = \text{Cov}(f(x), f(x'))$$

El GP queda definido por su media $\mu$ y su kernel $k$.

## 3. El kernel (covariance function)

El kernel codifica nuestras creencias sobre la función: suavidad, periodicidad, estacionariedad.

### 3.1 RBF (Radial Basis Function)

$$k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

- $\ell$ (length scale): cuán rápido varía la función. $\ell=0.1$ → muy irregular. $\ell=10$ → muy suave.
- $\sigma^2$: escala vertical (cuánto se desvía de la media).

### 3.2 Matern

$$k(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} d}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} d}{\ell}\right)$$

- $\nu=1.5$: funciones diferenciables una vez (menos suave que RBF)
- $\nu=2.5$: diferenciable dos veces (recomendado)

### 3.3 Periodic

$$k(x, x') = \sigma^2 \exp\left(-\frac{2 \sin^2(\pi |x - x'|/p)}{\ell^2}\right)$$

Para datos periódicos (temperatura anual, tráfico diario).

**Combinación de kernels**: se pueden sumar o multiplicar.
$$k = k_{\text{RBF}} + k_{\text{Periodic}}$$ captura tendencia suave + periodicidad.

## 4. GP Regression paso a paso

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# 1. Datos simulados: f(x) = sin(x) + ruido
X_train = np.random.uniform(0, 10, 20).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 20)

# 2. Definir kernel: RBF + ruido
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# 3. Entrenar (aprende ℓ, σ², σ_noise)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train)

print(f"Kernel aprendido: {gp.kernel_}")
# Ejemplo: RBF(length_scale=1.2) + WhiteKernel(noise_level=0.08)

# 4. Predecir con incertidumbre
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_mean, y_std = gp.predict(X_test, return_std=True)

# y_mean: predicción
# y_std: desviación estándar (incertidumbre)
```

**Salida esperada:**
```
Kernel aprendido: RBF(length_scale=1.2) + WhiteKernel(noise_level=0.08)
```

La incertidumbre ($y_{std}$) es baja cerca de los datos y crece en zonas sin observaciones.

## 5. Log Marginal Likelihood

Los hiperparámetros ($\ell, \sigma^2, \sigma_n^2$) se aprenden maximizando [[Linear Algebra|la verosimilitud marginal]]:

$$\log p(y | X) = -\frac{1}{2} y^T (K + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{n}{2} \log 2\pi$$

**Interpretación**:
- Primer término: qué tan bien se ajusta a los datos
- Segundo término: penaliza modelos complejos (determinante grande)
- Es una **regularización automática** — el GP balancea fit y complejidad

## 6. GP Classification

Para clasificación binaria, pasamos el GP por una sigmoide:

$$p(y = 1 | x) = \sigma(f(x)), \quad f \sim \mathcal{GP}(0, k)$$

No hay solución cerrada (la sigmoide rompe la Gaussianidad). Se usa aproximación de Laplace o inferencia variacional.

```python
from sklearn.gaussian_process import GaussianProcessClassifier

gpc = GaussianProcessClassifier(kernel=1.0 * RBF(), n_restarts_optimizer=5)
gpc.fit(X_train, y_train)
```

## 7. Sparse GPs (para big data)

GP inference es $\mathcal{O}(n^3)$. Para $n > 10,000$, es impracticable. Sparse GPs usan $m \ll n$ puntos inducing.

| Método | Complejidad | Uso |
|--------|------------|-----|
| Full GP | $\mathcal{O}(n^3)$ | n < 5,000 |
| FITC (sparse) | $\mathcal{O}(nm^2)$ | n < 100,000 |
| KISS-GP | $\mathcal{O}(n)$ | Grid estructurado |

## 8. Aplicaciones

### Bayesian Optimization

El uso más popular: optimizar funciones caras (hyperparameter tuning, diseño de experimentos):

1. GP surrogate de la función objetivo
2. Acquisition function (Expected Improvement) elige el próximo punto a evaluar
3. Evaluar, actualizar GP, repetir

```python
from skopt import gp_minimize

res = gp_minimize(func, dimensions, n_calls=30, acq_func='EI')
# GP internamente modela func, EI decide el próximo punto
```

## 9. Common Mistakes

1. **No escalar inputs**: GPs son sensibles a la escala. Estandarizar X a media 0, var 1.
2. **Kernel RBF para todo**: RBF asuma suavidad infinita. Para datos reales, Matern-5/2 es mejor.
3. **Ignorar incertidumbre**: el GP da $\sigma$ por un motivo. Usalo para active learning, Bayesian optimization, o para saber dónde no confiar.
4. **$\mathcal{O}(n^3)$ sorpresa**: GP con 50,000 puntos full = inviable. Usá sparse GP o aproximaciones.

## 10. Check Your Understanding

1. Predecís con un GP en $x=50$, lejos de todo los datos. ¿Qué esperás que pase con la media y la varianza?
2. $\ell$ (length scale) aprendido = 0.01. ¿Qué te dice sobre la función?
3. ¿Por qué GP usa maximización de la verosimilitud marginal en vez de cross-validation?

**Respuestas rápidas:**
1. Media → 0 (prior mean), varianza → $\sigma^2$ (prior variance). El GP "no sabe" y vuelve al prior.
2. La función es muy irregular (varía rápido). El modelo necesita muchos datos para ser preciso.
3. La verosimilitud marginal penaliza automáticamente la complejidad (regularización), CV no es necesario.

## 11. Summary

Gaussian Processes son distribuciones sobre funciones definidas por un kernel. Hacen regresión no paramétrica con incertidumbre calibrada. El kernel codifica suposiciones (RBF: suavidad, Periodic: ciclos). Los hiperparámetros se aprenden por máxima verosimilitud marginal. La limitación principal es $\mathcal{O}(n^3)$. Se usan en Bayesian Optimization, modelado de funciones, y active learning.

## 12. Where to Go Next

- [[Bayesian Inference]] — fundamento teórico
- [[Probabilistic Programming]] — GPs con PyMC
- [[Hyperparameter Tuning]] — Bayesian Optimization con GP surrogate
- [[Markov Chain Monte Carlo]] — métodos alternativos de inferencia
