---
tags:
  - statistics
  - experimental-design
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu equipo lanza una nueva feature en el checkout y ves que las métricas subieron 5%. ¿Fue por tu cambio o por casualidad? Un diseño experimental riguroso te permite responder con confianza estadística.

---

## 1. Hipótesis nula y alternativa

La **hipótesis nula (H₀)** asume que no hay efecto: la nueva feature no cambió la conversión. La **hipótesis alternativa (H₁)** es lo que quieres demostrar: la feature sí cambió la conversión.

**One-tailed** (una cola): H₁: μ_nuevo > μ_control (solo te importa si sube). **Two-tailed** (dos colas): H₁: μ_nuevo ≠ μ_control (te importa cualquier cambio).

```python
# plantilla conceptual
from scipy import stats

# supongamos datos simulados
control = [0.12, 0.11, 0.13, 0.10, 0.12]
tratamiento = [0.15, 0.16, 0.14, 0.17, 0.15]

t_stat, p_valor = stats.ttest_ind(control, tratamiento)
print(f"p-valor (two-tailed): {p_valor:.4f}")
```

**Salida esperada:**
```
p-valor (two-tailed): 0.0031
```

---

## 2. Error tipo I (α) y tipo II (β)

- **Error tipo I (α):** rechazar H₀ cuando es verdadera (falso positivo). Se fija típicamente en 0.05.
- **Error tipo II (β):** no rechazar H₀ cuando es falsa (falso negativo).
- **Poder estadístico (1 − β):** probabilidad de detectar un efecto real. Mínimo aceptable: 0.80.

| | H₀ verdadera | H₀ falsa |
|---|---|---|
| Rechazar H₀ | Error tipo I (α) | Acierto (1−β) |
| No rechazar H₀ | Acierto (1−α) | Error tipo II (β) |

---

## 3. Power analysis

El power analysis calcula el tamaño de muestra mínimo necesario para detectar un efecto de cierto tamaño con α y β dados.

```python
from statsmodels.stats.power import TTestIndPower

power_analysis = TTestIndPower()
tamano_muestra = power_analysis.solve_power(
    effect_size=0.3,   # effect size (d de Cohen)
    alpha=0.05,
    power=0.80,
    alternative='two-sided'
)
print(f"Tamaño de muestra por grupo: {tamano_muestra:.0f}")
```

**Salida esperada:**
```
Tamaño de muestra por grupo: 175
```

---

## 4. Randomization

La **asignación aleatoria** asegura que los grupos sean comparables en promedio. **Blocking** agrupa unidades similares y asigna tratamientos dentro de cada bloque (reduce varianza). **Stratified randomization** estratifica por variables clave (país, dispositivo) y aleatoriza dentro de cada estrato.

```python
import numpy as np
usuarios = np.arange(1000)
np.random.shuffle(usuarios)
control = usuarios[:500]
tratamiento = usuarios[500:]
```

---

## 5. Factorial designs

Permiten probar múltiples variables simultáneamente (ej: precio × color del botón) y detectar **interacciones**: el efecto de una variable depende del nivel de otra.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# datos simulados con interacción
# formula: y ~ precio * color
```

---

## 6. Sample size: ejemplo concreto

Tu app tiene 50 000 usuarios diarios. La conversión base es 5% y esperas subir a 6% (effect size absoluto de 1 p.p., d de Cohen ≈ 0.05). Necesitas calcular cuántos usuarios incluir en el experimento.

```python
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower

es = proportion_effectsize(0.05, 0.06)
n = NormalIndPower().solve_power(es, alpha=0.05, power=0.80, alternative='two-sided')
print(f"Usuarios por grupo: {n:.0f}")
```

**Salida esperada:**
```
Usuarios por grupo: 7348
```

---

## 7. Common Mistakes

1. **Peeking:** mirar los resultados antes de que termine el experimento e inflar α.
2. **Early stopping:** detener el experimento porque el p-valor es significativo temprano.
3. **Múltiples tests sin corrección:** comparar 10 métricas con α=0.05 da ~50% de chance de al menos un falso positivo. Usar Bonferroni o FDR.

```python
# Corrección Bonferroni
from statsmodels.stats.multitest import multipletests
p_valores = [0.01, 0.04, 0.20, 0.50, 0.03]
rechazar, p_corregidos, _, _ = multipletests(p_valores, method='bonferroni')
print(p_corregidos)
```

**Salida esperada:**
```
[0.05 0.2  1.   1.   0.15]
```

---

## Resumen

1. Define H₀ y H₁ antes de recolectar datos.
2. Fija α (típicamente 0.05) y busca poder ≥ 0.80.
3. Usa power analysis para determinar el tamaño de muestra.
4. Aleatoriza correctamente — blocking y stratified randomization mejoran precisión.
5. Los diseños factoriales permiten estudiar interacciones entre variables.
6. No mires los datos antes de tiempo ni detengas experimentos prematuramente.
7. Corrige por múltiples comparaciones cuando evalúes varias métricas.

---

## Check Your Understanding

1. Si corres 20 tests con α=0.05, ¿cuántos falsos positivos esperas en promedio? <!-- 1 (20 × 0.05) -->
2. ¿Qué significa poder estadístico de 0.80? <!-- 80% de probabilidad de detectar un efecto real -->
3. ¿Por qué el peeking infla la tasa de error tipo I? <!-- Cada vez que miras tienes chance de parar por azar, acumulando α -->
4. ¿Qué ventaja tiene un diseño factorial sobre probar cada variable por separado? <!-- Permite detectar interacciones entre variables -->

---

## Where to Go Next

- [[A-B Testing]]
- [[Causal Inference]]
- [[Resampling & Bootstrap]]
- [[Model Evaluation]]
