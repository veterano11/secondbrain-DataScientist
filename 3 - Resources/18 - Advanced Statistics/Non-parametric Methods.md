---
tags:
  - statistics
  - non-parametric
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tus datos no son normales, la muestra es pequeña, y no puedes usar tests paramétricos. Los métodos no paramétricos no asumen distribuciones específicas y funcionan con rankings en lugar de valores.

---

## 1. ¿Cuándo usar no paramétricos?

Usa métodos no paramétricos cuando:
- Muestra pequeña (n < 30) sin garantía de normalidad.
- Datos ordinales o con outliers severos.
- La variable no cumple supuestos de homogeneidad de varianza.

```python
# Test de normalidad previo
from scipy.stats import shapiro, anderson

datos = [2.1, 3.0, 3.2, 4.5, 5.1, 6.0, 8.5, 12.0, 1.2, 1.5]
stat, p = shapiro(datos)
print(f"Shapiro-Wilk p-valor: {p:.3f}")
```

**Salida esperada:**
```
Shapiro-Wilk p-valor: 0.042
```

p < 0.05 → los datos no son normales → prefiere un test no paramétrico.

---

## 2. Mann-Whitney U test

Alternativa no paramétrica al **t-test independiente**. Compara si una población tiende a tener valores mayores que la otra, basándose en los rankings.

```python
from scipy.stats import mannwhitneyu

grupo_a = [1.5, 2.0, 2.5, 3.0, 3.5]
grupo_b = [4.0, 4.5, 5.0, 5.5, 6.0]

stat, p_valor = mannwhitneyu(grupo_a, grupo_b, alternative='two-sided')
print(f"U = {stat:.0f}, p-valor = {p_valor:.4f}")
```

**Salida esperada:**
```
U = 0, p-valor = 0.0079
```

Un U de 0 indica que todos los valores de un grupo son mayores que los del otro — la diferencia es significativa.

---

## 3. Wilcoxon signed-rank test

Alternativa al **t-test pareado** (muestras dependientes: antes/después, matching). Opera sobre las diferencias entre pares.

```python
from scipy.stats import wilcoxon

antes = [8, 10, 12, 9, 11, 7, 13]
despues = [12, 14, 15, 10, 16, 11, 18]

stat, p_valor = wilcoxon(antes, despues, alternative='two-sided')
print(f"W = {stat:.0f}, p-valor = {p_valor:.4f}")
```

**Salida esperada:**
```
W = 0, p-valor = 0.0156
```

---

## 4. Kruskal-Wallis H test

ANOVA no paramétrico para comparar **3 o más grupos independientes**. Extiende Mann-Whitney a múltiples grupos.

```python
from scipy.stats import kruskal

grupo_c = [1.2, 1.5, 1.8, 2.0, 2.2]
grupo_d = [2.5, 3.0, 3.2, 3.5, 4.0]
grupo_e = [4.5, 5.0, 5.5, 6.0, 6.5]

stat, p_valor = kruskal(grupo_c, grupo_d, grupo_e)
print(f"H = {stat:.2f}, p-valor = {p_valor:.4f}")
```

**Salida esperada:**
```
H = 12.38, p-valor = 0.0021
```

Si el test es significativo, usa **Dunn's post-hoc** para identificar qué pares difieren.

---

## 5. Spearman rank correlation

Correlación basada en **rangos**, no en valores. Es robusta a outliers y no asume relación lineal.

```python
from scipy.stats import spearmanr

horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
puntaje = [2, 4, 3, 6, 8, 7, 9, 10, 1, 5]
# Outlier evidente en la última observación

coef, p_valor = spearmanr(horas_estudio, puntaje)
print(f"Spearman ρ = {coef:.3f}, p-valor = {p_valor:.3f}")
```

**Salida esperada:**
```
Spearman ρ = 0.455, p-valor = 0.187
```

Comparado con Pearson (que sería ~0.26 aquí), Spearman es menos sensible al outlier.

---

## 6. Power y eficiencia

Los tests no paramétricos tienen **menor poder** que sus equivalentes paramétricos cuando los supuestos paramétricos se cumplen. La **eficiencia relativa** es ~95% para Mann-Whitney vs t-test con normales.

| Situación | Recomendación |
|---|---|
| Datos normales, varianzas iguales | t-test (más poder) |
| Datos no normales o muestra pequeña | Mann-Whitney / Wilcoxon |
| Outliers severos | No paramétrico (robusto) |
| Datos ordinales | No paramétrico (obligatorio) |

---

## 7. Common Mistakes

1. **Usar Mann-Whitney para datos pareados**: debe ser Wilcoxon signed-rank.
2. **Ignorar empates en rankings**: empates reducen la precisión — usa corrección por empates (scipy la aplica automáticamente).
3. **No ajustar por comparaciones múltiples en Kruskal-Wallis post-hoc**: usa Dunn's test con corrección.

```python
from scipy.stats import mannwhitneyu

# CORRECTO: grupos independientes
mannwhitneyu(grupo_a, grupo_b)

# INCORRECTO: si los datos son pareados (antes/después)
# Usar wilcoxon() en su lugar
```

---

## Resumen

1. Usa tests no paramétricos con muestras pequeñas, no normales u ordinales.
2. Mann-Whitney U para dos grupos independientes; Wilcoxon para pareados.
3. Kruskal-Wallis para 3+ grupos independientes (ANOVA no paramétrico).
4. Spearman correlation es robusta a outliers basada en rankings.
5. Los no paramétricos pierden poco poder frente a paramétricos bajo normalidad (~5%).
6. No confundas tests para datos independientes vs pareados.

---

## Check Your Understanding

1. ¿Cuándo usarías Wilcoxon signed-rank en lugar de Mann-Whitney? <!-- Cuando los datos son pareados (antes/después, mismo sujeto) -->
2. ¿Qué hace Kruskal-Wallis? <!-- Extiende Mann-Whitney a 3+ grupos independientes -->
3. ¿Por qué Spearman es más robusto a outliers que Pearson? <!-- Opera sobre rangos, no sobre valores absolutos -->
4. ¿Cuánto poder pierdes al usar Mann-Whitney en lugar de t-test con datos normales? <!-- ~5% de eficiencia relativa -->

---

## Where to Go Next

- [[Resampling & Bootstrap]]
- [[Statistics]]
- [[Experimental Design]]
- [[Model Evaluation]]
- [[Python for Data Science]]
