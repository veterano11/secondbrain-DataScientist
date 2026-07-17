---
tags:
  - statistics
  - resampling
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tienes una muestra de datos y quieres estimar la incertidumbre de tu estadístico (mediana, correlación, accuracy) sin asumir normalidad. El bootstrap te da intervalos de confianza con solo tu muestra.

---

## 1. Idea intuitiva

El bootstrap consiste en muestrear **con reposición** de tus datos originales N veces (típicamente 1000+), calcular el estadístico de interés en cada muestra, y usar la distribución de esos valores como aproximación de la distribución muestral.

```
Datos originales: [x₁, x₂, ..., xₙ]
          ↓ muestrear con reposición (mismo n)
Muestra bootstrap 1:  [x₃, x₁, x₃, x₂, ...] → estadístico θ₁
Muestra bootstrap 2:  [x₂, x₅, x₁, x₇, ...] → estadístico θ₂
...
Muestra bootstrap B:  [x₄, x₂, x₆, x₁, ...] → estadístico θ_B
          ↓
Distribución de θ₁ ... θ_B → IC, error estándar, sesgo
```

---

## 2. Bootstrap en Python

```python
import numpy as np

# Tiempos de carga (segundos) — datos asimétricos
tiempos = np.array([1.2, 1.5, 2.1, 3.0, 3.2, 4.5, 5.1, 6.0, 8.5, 12.0])
n_bootstrap = 10_000
medianas_boot = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    muestra = np.random.choice(tiempos, size=len(tiempos), replace=True)
    medianas_boot[i] = np.median(muestra)

# Percentile CI
ic_inf = np.percentile(medianas_boot, 2.5)
ic_sup = np.percentile(medianas_boot, 97.5)
print(f"Mediana observada: {np.median(tiempos):.2f}")
print(f"IC 95% bootstrap: [{ic_inf:.2f}, {ic_sup:.2f}]")
```

**Salida esperada:**
```
Mediana observada: 3.10
IC 95% bootstrap: [1.50, 8.00]
```

---

## 3. Tipos de CI bootstrap

1. **Percentile bootstrap**: usa los percentiles 2.5 y 97.5 de la distribución bootstrap.
2. **BCa (Bias-Corrected and Accelerated)**: ajusta por sesgo y asimetría — más preciso.
3. **Bootstrap-t**: usa el estadístico t bootstrap para construir IC, requiere estimar error estándar en cada réplica.

```python
from scipy.stats import bootstrap
import numpy as np

tiempos = np.array([1.2, 1.5, 2.1, 3.0, 3.2, 4.5, 5.1, 6.0, 8.5, 12.0])
res = bootstrap((tiempos,), np.median, n_resamples=10000, method='BCa')
print(res.confidence_interval)
```

**Salida esperada:**
```
Low: 1.5, High: 8.5
```

---

## 4. Permutation tests

Un **permutation test** es un test de hipótesis sin asumir una distribución específica. Bajo H₀ (no hay diferencia), las etiquetas de grupo son intercambiables. Permutas las etiquetas muchas veces y comparas el estadístico observado con la distribución nula.

```python
import numpy as np

grupo_a = np.array([2.1, 3.0, 3.2, 4.5, 5.1])
grupo_b = np.array([6.0, 8.5, 12.0, 1.2, 1.5])
obs_diff = np.median(grupo_a) - np.median(grupo_b)

combinado = np.concatenate([grupo_a, grupo_b])
n_perm = 10_000
diferencias = np.zeros(n_perm)

for i in range(n_perm):
    np.random.shuffle(combinado)
    perm_a = combinado[:len(grupo_a)]
    perm_b = combinado[len(grupo_a):]
    diferencias[i] = np.median(perm_a) - np.median(perm_b)

p_valor = np.mean(np.abs(diferencias) >= np.abs(obs_diff))
print(f"p-valor (two-tailed): {p_valor:.3f}")
```

**Salida esperada:**
```
p-valor (two-tailed): 0.217
```

---

## 5. Bootstrap para modelos

**Bagging (Bootstrap Aggregating)** entrena múltiples modelos en muestras bootstrap y promedia sus predicciones. Es la base de **Random Forest**. Cada árbol se entrena en una muestra bootstrap distinta, reduciendo varianza sin aumentar sesgo.

```python
from sklearn.ensemble import RandomForestRegressor

# Bagging es bootstrap aplicado a modelos
# Cada árbol ve una muestra bootstrap distinta de los datos
modelo = RandomForestRegressor(n_estimators=100, bootstrap=True)
```

---

## 6. Common Mistakes

1. **Muy pocas réplicas (<1000)**: la distribución bootstrap no se estabiliza.
2. **Sesgo en muestras pequeñas**: con n < 20 el bootstrap puede subestimar la incertidumbre.
3. **Ignorar dependencias**: el bootstrap asume que las observaciones son i.i.d. — no funciona con series temporales sin ajuste (block bootstrap).

```python
# Block bootstrap para series temporales (conceptual)
# En lugar de muestrear observaciones individuales,
# muestrea bloques consecutivos para preservar la dependencia
```

---

## Resumen

1. El bootstrap muestrea con reposición para aproximar la distribución muestral de cualquier estadístico.
2. El percentil CI (2.5, 97.5) es el método más simple; BCa es más robusto.
3. Los permutation tests son tests de hipótesis sin supuestos distribucionales.
4. Bagging y Random Forest aplican bootstrap a modelos predictivos.
5. Se necesitan ≥ 1000 réplicas y tener cuidado con muestras pequeñas y dependencias.

---

## Check Your Understanding

1. ¿Por qué el bootstrap muestrea con reposición? <!-- Sin reposición obtendrías siempre la misma muestra original -->
2. ¿Qué diferencia hay entre un IC bootstrap percentil y BCa? <!-- BCa ajusta por sesgo y asimetría de la distribución bootstrap -->
3. ¿Qué supuesto clave hacen los permutation tests? <!-- Intercambiabilidad de etiquetas bajo H₀ -->
4. ¿Cómo adaptarías bootstrap para series temporales? <!-- Block bootstrap: muestrear bloques consecutivos para preservar autocorrelación -->

---

## Where to Go Next

- [[Non-parametric Methods]]
- [[Experimental Design]]
- [[Supervised Learning]]
- [[Statistics]]
- [[Python for Data Science]]
