---
tags:
  - seaborn
  - statistics
  - EDA
status: seedling
created: 2026-06-28
---

# seaborn & Statistical Plots

## Escenario de aprendizaje

Tienes un dataset de estudiantes con horas de estudio, calificaciones, género y grupo socioeconómico. Necesitas explorar relaciones, distribuciones y diferencias entre grupos en una sesión de EDA (Exploratory Data Analysis). seaborn hace en 1 línea lo que matplotlib hace en 10, y además calcula estadísticos automáticamente.

El objetivo es entender qué factores se correlacionan con mayor rendimiento académico antes de construir un modelo predictivo.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset simulado (inspirado en Students Performance de Kaggle)
np.random.seed(42)
n = 500
data = {
    'horas_estudio': np.random.exponential(5, n).clip(0, 25),
    'calificacion': np.random.normal(70, 12, n).clip(0, 100),
    'genero': np.random.choice(['M', 'F'], n),
    'grupo_socioeco': np.random.choice(['Bajo', 'Medio', 'Alto'], n, p=[0.3, 0.5, 0.2]),
    'asistencia': np.random.uniform(50, 100, n)
}
df = pd.DataFrame(data)
# Añadir correlación artificial: +0.3 entre horas_estudio y calificacion
df['calificacion'] += df['horas_estudio'] * 1.5
df['calificacion'] = df['calificacion'].clip(0, 100)

print(df.head())
print(df.describe())
```

## 1. seaborn vs matplotlib

seaborn está construido sobre matplotlib. Sus ventajas principales:

- **API integrada con pandas**: pasas el DataFrame directamente con `data=df`.
- **Themas profesionales**: `sns.set_theme()` aplica estilos listos para publicación.
- **Estadísticos automáticos**: boxplots muestran medianas, IQR, outliers; regplot calcula regresión lineal.
- **Facetas**: dividir datos por categorías en una sola línea (`col=`, `row=`, `hue=`).

```python
sns.set_theme(style='whitegrid', palette='Set2')

# matplotlib: ~10 líneas
fig, ax = plt.subplots()
ax.scatter(df['horas_estudio'], df['calificacion'], alpha=0.5)
ax.set_xlabel('Horas de estudio'); ax.set_ylabel('Calificación')

# seaborn: 1 línea
sns.scatterplot(data=df, x='horas_estudio', y='calificacion')
# Salida esperada: dos scatter plots similares, pero seaborn trae tema profesional por defecto
```

Para más principios de selección visual, revisa [[Visualization Fundamentals]].

## 2. Distribuciones

Entender cómo se distribuyen las variables es el paso #1 del EDA.

| Función | Uso |
|---|---|
| `histplot` | Histograma con control de bins |
| `kdeplot` | Kernel Density Estimate (curva suave) |
| `displot` | histplot + kdeplot combinado (Figure-level) |
| `rugplot` | Marca cada observación en el eje |

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Histograma simple
sns.histplot(data=df, x='calificacion', bins=25, ax=axes[0, 0])
axes[0, 0].set_title('Histograma de calificaciones')

# KDE
sns.kdeplot(data=df, x='calificacion', fill=True, ax=axes[0, 1])
axes[0, 1].set_title('Densidad (KDE)')

# Histograma + KDE + rug
sns.histplot(data=df, x='horas_estudio', kde=True, rug=True, bins=30, ax=axes[1, 0])
axes[1, 0].set_title('Histograma + KDE + Rug')

# displot (Figure-level): muestra joint + marginal
g = sns.displot(data=df, x='horas_estudio', y='calificacion', kind='kde')
g.fig.suptitle('Distribución conjunta: horas vs calificación', y=1.02)

plt.tight_layout()
# Salida esperada: 3 subplots + 1 displot bivariado. Se observa sesgo derecho en horas_estudio
```

## 3. Relaciones categóricas

Cuando quieres comparar distribuciones numéricas entre grupos, seaborn ofrece visualizaciones informativas.

| Función | Cuándo usarla |
|---|---|
| `boxplot` | Resumen estadístico (mediana, IQR, outliers) |
| `violinplot` | Boxplot + KDE (forma de la distribución) |
| `stripplot` | Todos los puntos (útil con jitter) |
| `swarmplot` | Puntos que no se solapan (muestras pequeñas) |
| `barplot` | Media + intervalo de confianza |

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Boxplot
sns.boxplot(data=df, x='genero', y='calificacion', hue='genero', ax=axes[0],
            palette='Set2', legend=False)
axes[0].set_title('Boxplot: calificación por género')

# Violinplot
sns.violinplot(data=df, x='grupo_socioeco', y='calificacion', hue='grupo_socioeco',
               ax=axes[1], palette='Set2', inner='quartile', legend=False)
axes[1].set_title('Violinplot: calificación por grupo socioeconómico')

# Swarmplot + Boxplot combinados (muestra pequeña)
df_sample = df.sample(60)
sns.boxplot(data=df_sample, x='genero', y='calificacion', hue='genero',
            ax=axes[2], palette='Set2', legend=False, width=0.3)
sns.swarmplot(data=df_sample, x='genero', y='calificacion', hue='genero',
              ax=axes[2], color='black', alpha=0.7, legend=False)
axes[2].set_title('Box + Swarm (n=60)')

plt.tight_layout()
# Salida esperada: 3 paneles — boxplot (género), violinplot (grupo socioeco), box+swarm combinados
```

**Interpretación**: si las cajas no se solapan o las diferencias de medias son > 2× el IQR, probablemente hay diferencia significativa. [[Feature Engineering]] usa estos insights para crear variables.

## 4. Relaciones numéricas

Para explorar relaciones entre variables continuas, seaborn añade automáticamente líneas de regresión y segmentación por categorías.

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatterplot simple
sns.scatterplot(data=df, x='horas_estudio', y='calificacion', alpha=0.5, ax=axes[0, 0])
axes[0, 0].set_title('Scatter: horas vs calificación')

# Scatter con hue (color por género)
sns.scatterplot(data=df, x='horas_estudio', y='calificacion', hue='genero',
                alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('Separado por género')

# Scatter + regresión lineal
sns.regplot(data=df, x='horas_estudio', y='calificacion', scatter_kws={'alpha': 0.4},
            line_kws={'color': '#C44E52'}, ax=axes[1, 0])
axes[1, 0].set_title('Regresión: tendencia lineal')

# lmplot con hue y facetas (Figure-level)
g = sns.lmplot(data=df, x='horas_estudio', y='calificacion', col='grupo_socioeco',
               hue='genero', height=3.5, facet_kws={'sharex': True, 'sharey': True})
g.fig.suptitle('Regresión por grupo socioeconómico y género', y=1.02)
# Salida esperada: scatter, scatter+hue, regplot con línea, lmplot con facetas 3×2
```

**Atención**: `regplot` (Axes-level) vs `lmplot` (Figure-level con facetas). Elige `regplot` para un solo panel. [[Interactive Visualization (plotly-altair)]] permite hover interactivo sobre los puntos.

## 5. Matrices de correlación

Las matrices de correlación resumen relaciones entre todas las variables numéricas. seaborn las visualiza con heatmaps, pairplots y clustermaps.

```python
# Heatmap de correlación
corr = df.select_dtypes(include=[np.number]).corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1,
            square=True, ax=axes[0])
axes[0].set_title('Matriz de correlación')

# Clustermap: heatmap + dendrograma (agrupa variables correlacionadas)
g = sns.clustermap(corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1,
                   figsize=(6, 5), linewidths=0.5)
g.fig.suptitle('Clustermap: variables agrupadas por similitud', y=1.02)
# Salida esperada: heatmap cuadrado con valores, clustermap con dendrogramas en bordes
```

**Pairplot**: scatter matrix de todas las variables. Útil para detectar relaciones no lineales. Precaución: con muchas variables se vuelve ilegible (ver Common Mistakes).

```python
sns.pairplot(df, hue='genero', corner=True, height=2.5)
# Salida esperada: matriz de scatter plots con diagonal = KDE, coloreado por género
```

## 6. Estilos y temas

seaborn permite cambiar la apariencia global con `set_theme()` y personalizar paletas.

```python
# Temas predefinidos
# sns.set_theme(style='whitegrid')   # Fondo blanco con grid
# sns.set_theme(style='darkgrid')    # Fondo gris oscuro con grid
# sns.set_theme(style='ticks')       # Ejes con marcas, sin grid

# Paletas personalizadas
palette_custom = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
sns.set_palette(palette_custom)

# Context: ajusta escala para diferentes medios
# sns.set_context('paper')    # Tamaños pequeños (artículo)
# sns.set_context('notebook') # Tamaño estándar
# sns.set_context('talk')     # Grande (presentación)
# sns.set_context('poster')   # Muy grande (póster)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df, x='grupo_socioeco', y='calificacion', hue='grupo_socioeco',
            ax=ax, legend=False)
ax.set_title('Barplot con paleta personalizada')
# Salida esperada: barplot con colores personalizados seteados globalmente
```

## 7. Facetas: dividir y conquistar

Las facetas crean múltiples paneles automáticamente según variables categóricas. Es la forma más rápida de explorar interacciones.

| Función | Tipo | Parámetros clave |
|---|---|---|
| `FacetGrid` | Clase base | `col`, `row`, `hue` |
| `catplot` | Figure-level | `kind='box'|'violin'|'bar'|'point'` |
| `lmplot` | Figure-level | `col`, `row`, `hue` + regresión |
| `displot` | Figure-level | `kind='hist'|'kde'|'ecdf'` |

```python
# FacetGrid manual
g = sns.FacetGrid(df, col='grupo_socioeco', row='genero', height=3, margin_titles=True)
g.map(sns.scatterplot, 'horas_estudio', 'calificacion', alpha=0.6)
g.fig.suptitle('FacetGrid: horas vs calificación por grupo y género', y=1.02)
# Salida esperada: grid 2×3 (2 géneros × 3 grupos) con scatter plots

# catplot: boxplots facetados
g = sns.catplot(data=df, x='genero', y='calificacion', col='grupo_socioeco',
                kind='box', height=4)
g.fig.suptitle('Boxplots facetados por grupo socioeconómico', y=1.02)
# Salida esperada: 3 paneles (uno por grupo) con boxplots M vs F
```

[[Python for Data Science]] muestra cómo integrar facetas en pipelines de análisis automatizados.

## 8. Common Mistakes

| Error | Problema | Solución |
|---|---|---|
| Pairplot con >6 variables | Matriz gigante e ilegible | Usar `corner=True` o seleccionar columnas clave |
| Malinterpretar outliers | Confundir outliers con errores | Outliers en boxplot son puntos > 1.5×IQR; no siempre son errores |
| Olvidar `hue` | No separar grupos y perder patrones | Siempre preguntar: "¿hay una variable categórica relevante?" |
| Ignorar el tamaño muestral | Violinplots con n < 20 son engañosos | Usar swarmplot para muestras pequeñas |
| No ajustar `kde_bw` | KDE demasiado suave o con picos | Ajustar `bw_adjust` o probar `cut=0` |

```python
# ❌ Pairplot con demasiadas variables
df_largo = df.copy()
for i in range(8):
    df_largo[f'ruido_{i}'] = np.random.randn(500)
# sns.pairplot(df_largo)  # 11×11 = 121 paneles → ilegible

# ✅ Pairplot con corner=True y columnas relevantes
sns.pairplot(df[['horas_estudio', 'calificacion', 'asistencia', 'genero']],
             hue='genero', corner=True)
# Salida esperada: matriz triangular inferior con scatter plots + diagonal KDE
```

## Resumen

1. seaborn se integra con pandas vía `data=df` y ofrece temas profesionales con `sns.set_theme()`.
2. `histplot`, `kdeplot` y `displot` exploran distribuciones univariadas y bivariadas.
3. `boxplot`, `violinplot`, `swarmplot` comparan distribuciones entre grupos categóricos.
4. `scatterplot` + `regplot` exploran relaciones numéricas; `lmplot` añade facetas y regresión.
5. `heatmap`, `pairplot` y `clustermap` resumen matrices de correlación multivariadas.
6. Los temas se controlan con `set_theme()`, paletas personalizadas y contextos (paper/notebook/talk/poster).
7. Las facetas (`catplot`, `lmplot`, `FacetGrid`) dividen datos por categorías en paneles separados.
8. Errores comunes: pairplots sobredimensionados, malinterpretar outliers de boxplot, olvidar `hue`.

## Check Your Understanding

1. ¿Cuál es la diferencia entre `regplot` y `lmplot`?
   *Respuesta: `regplot` es Axes-level (un solo panel), `lmplot` es Figure-level y soporta facetas (col/row/hue) con regresión.*

2. ¿Qué hace el parámetro `hue` en seaborn?
   *Respuesta: Colorea los puntos/líneas según una variable categórica, permitiendo ver patrones por grupo sin separar en paneles.*

3. ¿Cuándo usarías `swarmplot` en lugar de `boxplot`?
   *Respuesta: Cuando el tamaño muestral es pequeño (<50) y quieres ver cada observación individual sin overplotting.*

4. ¿Qué significan los puntos fuera de los "bigotes" en un boxplot?
   *Respuesta: Son outliers definidos como puntos > 1.5×IQR del cuartil superior o < 1.5×IQR del cuartil inferior. No necesariamente son errores.*

5. ¿Por qué limitar el pairplot a 5-6 variables?
   *Respuesta: Porque la matriz tiene n×(n-1)/2 paneles; con 10 variables son 45 scatter plots, imposibles de interpretar.*

## Where to Go Next

- [[matplotlib en profundidad]]
- [[Visualization Fundamentals]]
- [[Interactive Visualization (plotly-altair)]]
- [[Python for Data Science]]
- [[Feature Engineering]]
