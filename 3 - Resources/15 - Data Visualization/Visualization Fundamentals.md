---
tags:
  - data-viz
  - fundamentals
status: seedling
created: 2026-06-28
---

# Visualization Fundamentals

## Escenario de aprendizaje

Tienes un dataset de ventas por región y necesitas presentarlo al equipo. ¿Gráfico de barras? ¿Pastel? ¿Líneas? Elegir mal el gráfico puede hacer que tu audiencia saque conclusiones equivocadas. Esta nota cubre los principios perceptivos y la guía de selección de gráficos para que comuniques datos con claridad.

Antes de escribir código, pregúntate: ¿qué quiero que mi audiencia vea primero? La respuesta define el canal visual adecuado.

## 1. Percepción visual: pre-attentive attributes

El ojo humano procesa ciertos atributos en paralelo (milisegundos) sin esfuerzo consciente. Son los **pre-attentive attributes** y son la base del diseño visual efectivo.

| Atributo | Uso típico | Efectividad |
|---|---|---|
| Posición | Lo más preciso para comparar valores | Alta |
| Longitud | Barras, ejes | Alta |
| Ángulo | Pie charts | Media |
| Área | Burbujas, mapas de calor | Baja |
| Color (tono) | Categorías | Alta (si pocas categorías) |
| Color (intensidad) | Variables continuas | Alta |
| Forma | Marcadores de scatter | Media |

```python
import matplotlib.pyplot as plt
import numpy as np

# Pre-attentive demo: encontrar el punto rojo entre círculos grises
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
colors = ['gray'] * 50
colors[23] = 'red'  # único punto rojo

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, c=colors, s=100)
ax.set_title("Pre-attentive: el punto rojo 'pop out'")
# Salida esperada: scatter plot con 49 puntos grises + 1 rojo que el ojo detecta al instante
```

## 2. Guía de selección de gráficos

Cada gráfico responde una pregunta distinta. Esta guía relaciona objetivo → gráfico recomendado.

| Objetivo | Gráfico | Eje X | Eje Y |
|---|---|---|---|
| Comparar categorías | Barras | Categórica | Numérica |
| Distribución | Histograma / KDE | Numérica bins | Frecuencia |
| Relación | Scatter | Numérica | Numérica |
| Composición | Stacked bar / área | Categórica o tiempo | Proporción |
| Evolución en el tiempo | Líneas | Temporal | Numérica |
| Correlación | Heatmap | 2 categóricas | Valor numérico |

```python
import pandas as pd
import matplotlib.pyplot as plt

# Simular ventas trimestrales por región
data = {
    'Region': ['Norte', 'Sur', 'Este', 'Oeste'],
    'Q1': [120, 95, 140, 110],
    'Q2': [135, 100, 150, 115],
    'Q3': [130, 110, 145, 120],
    'Q4': [150, 115, 160, 130]
}
df = pd.DataFrame(data).melt(id_vars='Region', var_name='Trimestre', value_name='Ventas')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Barras agrupadas: comparar regiones por trimestre
for i, t in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    subset = df[df['Trimestre'] == t]
    axes[0].bar(subset['Region'] + '-' + i*0.2, subset['Ventas'], width=0.2, label=t)
axes[0].set_title('Barras: comparar regiones')
axes[0].legend()

# Líneas: evolución temporal por región
for region in df['Region'].unique():
    sub = df[df['Region'] == region]
    axes[1].plot(['Q1','Q2','Q3','Q4'], sub['Ventas'], marker='o', label=region)
axes[1].set_title('Líneas: evolución temporal')
axes[1].legend()

# Barras apiladas: composición
pivot = df.pivot(index='Trimestre', columns='Region', values='Ventas')
pivot.plot(kind='bar', stacked=True, ax=axes[2])
axes[2].set_title('Stacked bar: composición')
axes[2].legend()

plt.tight_layout()
# Salida esperada: 3 subplots — barras agrupadas, líneas, barras apiladas
```

## 3. El problema de los pie charts

El ojo humano es pésimo estimando ángulos. Juzgamos longitudes con precisión, pero comparar sectores en un pie chart requiere esfuerzo cognitivo.

```python
# Comparación: barras vs pie
labels = ['A', 'B', 'C', 'D']
values = [30, 28, 22, 20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.pie(values, labels=labels, autopct='%1.1f%%')
ax1.set_title('Pie: ¿cuál es más grande?')

ax2.bar(labels, values, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
ax2.set_title('Barras: A > B > C > D (claro)')
ax2.set_ylabel('Valor')

plt.tight_layout()
# Salida esperada: pie chart vs barras — el ranking es obvio en barras, dudoso en pie
```

**Regla práctica**: si tienes más de 3 categorías o necesitas comparar valores similares, usa barras en lugar de pie. Revisa [[Python for Data Science]] para más principios de exploración.

## 4. Color theory

El color comunica antes que cualquier otro canal. Elegir bien evita confusión y hace que tus gráficos sean accesibles.

- **Paletas categóricas** (Set1, Set2, tab10): categorías discretas sin orden.
- **Paletas secuenciales** (Blues, Greens, viridis): valores ordenados de bajo a alto.
- **Paletas divergentes** (RdBu, PiYG): desviación respecto a un punto medio (cero, promedio).

```python
import matplotlib.colors as mcolors

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Categórica
cat_cmap = plt.get_cmap('Set2')
for i in range(6):
    axes[0].bar(i, 1, color=cat_cmap(i), width=0.7)
axes[0].set_title('Categórica (Set2)')

# Secuencial
seq_cmap = plt.get_cmap('Blues')
for i in range(10):
    axes[1].bar(i, 1, color=seq_cmap(i/10), width=0.7)
axes[1].set_title('Secuencial (Blues)')

# Divergente
div_cmap = plt.get_cmap('RdBu')
for i in range(10):
    axes[2].bar(i, 1, color=div_cmap(i/10), width=0.7)
axes[2].set_title('Divergente (RdBu)')

plt.tight_layout()
# Salida esperada: 3 barras horizontales mostrando cada tipo de paleta
```

**Colorblind-friendly**: las paletas `viridis`, `magma`, `cividis`, `Set2`, `Dark2` funcionan bien para daltonismo. Evita rojo-verde juntos. [[seaborn & Statistical Plots]] usa `colorblind` como tema por defecto.

## 5. Aspect ratio y scale

La proporción entre ejes y la escala numérica cambian drásticamente la historia que cuenta un gráfico. Tres conceptos críticos:

- **Deceptive axes**: truncar el eje Y exagera diferencias pequeñas.
- **Start at zero**: para barras, el eje Y debe empezar en cero (leyendo la longitud). Para líneas, no es obligatorio.
- **Lie factor** (Tufte): tamaño del efecto visual ÷ tamaño del efecto en datos.

```python
# Barras engañosas: mismo dato, distinta escala
valores = [100, 102, 98, 105, 103]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
ax1.bar(range(5), valores)
ax1.set_ylim(0, 120)
ax1.set_title('Correcto: y arranca en 0')

ax2.bar(range(5), valores)
ax2.set_ylim(95, 106)
ax2.set_title('Engañoso: pequeñas diferencias se ven enormes')

plt.tight_layout()
# Salida esperada: dos barras de datos idénticos — una parece estable, la otra volátil
```

**Lie factor** = tamaño percibido en gráfico / tamaño real en datos. Valores cercanos a 1 indican integridad visual.

## 6. Chart junk y data-ink ratio

Edward Tufte acuñó el **data-ink ratio**: proporción de tinta que representa datos vs tinta total del gráfico. Maximizarlo significa eliminar decoración innecesaria.

| ❌ Chart junk | ✅ Minimalismo |
|---|---|
| Fondo con textura | Fondo blanco o gris claro |
| 3D innecesario | 2D plano |
| Sombras y gradients | Color plano |
| Líneas de grid excesivas | Grid sutil o solo horizontal |
| Marcas superfluas | Puntos solo donde hay datos |

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Chart junk
x = np.arange(5)
y = [3, 5, 2, 7, 4]
ax1.bar(x, y, color='red', edgecolor='black', linewidth=2)
ax1.set_title('Chart junk')
ax1.set_facecolor('#EEEEEE')

# Data-ink ratio alto
ax2.bar(x, y, color='#4C72B0', width=0.6)
ax2.set_title('Data-ink alto')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

plt.tight_layout()
# Salida esperada: gráfico recargado vs gráfico minimalista con la misma información
```

Revisa [[Storytelling with Data]] para profundizar en limpieza visual aplicada.

## 7. Common Mistakes

| Error | Problema | Solución |
|---|---|---|
| 3D charts | Distorsiona percepción de valores | Usar 2D siempre |
| Dual axes engañosos | Dos escalas distintas en un mismo plot | Separar en subplots o normalizar |
| Demasiados colores | Saturación cognitiva, confusión | Máximo 6-8 colores, paleta coherente |
| Overplotting | Puntos que se tapan unos a otros | Transparencia (alpha), jitter, hexbin |
| No ordenar barras | Dificulta comparación visual | Ordenar ascendente o descendente |

```python
# Dual axes engañoso: correlación espuria por escalas diferentes
fig, ax1 = plt.subplots(figsize=(6, 3))
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May']
ventas = [120, 130, 125, 140, 135]
tickets = [45, 42, 40, 38, 35]

ax1.bar(meses, ventas, color='#4C72B0', alpha=0.7, label='Ventas ($)')
ax1.set_ylabel('Ventas ($)')
ax2 = ax1.twinx()
ax2.plot(meses, tickets, 'o-', color='#C44E52', label='Tickets')
ax2.set_ylabel('Tickets (#)')
fig.legend()
ax1.set_title('Dual axes engañoso: ¿tickets bajan mientras ventas suben?')
# Salida esperada: barras y línea superpuestos con escalas distintas en lados opuestos
```

## Resumen

1. Los pre-attentive attributes (posición, longitud, color) determinan qué canales visuales son más efectivos.
2. La selección del gráfico depende de la pregunta: comparar (barras), distribuir (histograma), relacionar (scatter), evolucionar (líneas).
3. Los pie charts son difíciles de leer — prefiere barras para comparar proporciones.
4. Usa paletas categóricas, secuenciales y divergentes según el tipo de variable; prefiere opciones colorblind-safe.
5. El aspect ratio y la escala numérica afectan la percepción — evita ejes truncados en barras y mide el lie factor.
6. Maximiza el data-ink ratio eliminando chart junk decorativo.
7. Errores comunes: 3D, dual axes engañosos, demasiados colores, overplotting, barras sin orden.

## Check Your Understanding

1. ¿Por qué los pie charts son menos efectivos que las barras para comparar proporciones?
   *Respuesta: Porque juzgamos longitudes con precisión pero los ángulos requieren esfuerzo cognitivo. Barras usan el canal de longitud (pre-attentive).*

2. ¿Cuándo es aceptable que el eje Y no empiece en cero?
   *Respuesta: En gráficos de líneas que muestran evolución temporal, donde la pendiente es más relevante que la altura absoluta. En barras NUNCA.*

3. ¿Qué tipo de paleta usarías para mostrar desviación de temperatura respecto al promedio histórico?
   *Respuesta: Divergente (RdBu o PiYG), con un punto medio neutro para el promedio y colores opuestos para arriba/abajo.*

4. ¿Cuál es el data-ink ratio y por qué es importante?
   *Respuesta: Proporción de tinta que representa datos vs tinta total. Maximizarlo fuerza a eliminar decoración que no aporta información.*

5. Nombra dos alternativas al color rojo-verde para gráficos accesibles a daltónicos.
   *Respuesta: Viridis (azul-amarillo) y Set2 (azul-naranja-morado).*

## Where to Go Next

- [[matplotlib en profundidad]]
- [[seaborn & Statistical Plots]]
- [[Interactive Visualization (plotly-altair)]]
- [[Storytelling with Data]]
- [[Python for Data Science]]
