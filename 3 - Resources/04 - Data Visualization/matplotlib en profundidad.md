---
tags:
  - matplotlib
  - python
  - plotting
status: seedling
created: 2026-06-28
---

# matplotlib en profundidad

## Escenario de aprendizaje

Necesitas un gráfico listo para publicar en un paper o reporte: con ejes personalizados, leyendas bien ubicadas, anotaciones, múltiples subplots. matplotlib puede hacerlo todo, pero su API es enorme. Esta nota cubre el 20% de matplotlib que resuelve el 80% de los problemas de visualización.

Tu jefe te pide una figura con 4 paneles que muestre: ventas totales, distribución por producto, relación precio-demanda, y crecimiento mensual. Todo en un solo PNG listo para el informe.

## 1. Arquitectura: Figure vs Axes

matplotlib tiene dos capas fundamentales. Confundirlas es la fuente #1 de errores.

- **Figure**: el lienzo completo. Contiene uno o más Axes, títulos globales, colorbars.
- **Axes**: el área donde se dibujan datos. Tiene ejes X/Y, ticks, labels. NO es "axes" plural de "axis".

```python
import matplotlib.pyplot as plt

# Crear Figure vacía y añadir Axes manualmente
fig = plt.figure(figsize=(8, 4))         # Figure: el canvas
ax = fig.add_subplot(111)                # Axes: el área de plot
ax.plot([1, 2, 3], [4, 5, 6])            # Dibuja en el Axes
ax.set_title('Un Axes dentro de una Figure')
fig.suptitle('Título global de la Figure')  # Pertenece a Figure, no a Axes
# Salida esperada: plot simple con título local y título global
```

**Analogía**: Figure es el canvas del pintor; Axes son cada lienzo individual pegado sobre él. Siempre trabajas con Axes; la Figure solo los contiene.

## 2. Figure y Axes: plt.subplots()

La función `plt.subplots()` es la puerta de entrada. Crea Figure + N Axes en una línea.

```python
import numpy as np

# (fig, ax) para un solo plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1, 2], [0, 1, 4])
ax.set_xlabel('Eje X'); ax.set_ylabel('Eje Y')
ax.set_title('Un solo Axes')

# (fig, axes) para grid de subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex='col', sharey='row')
for i, ax in enumerate(axes.flat):
    ax.plot(np.cumsum(np.random.randn(50)))
    ax.set_title(f'Serie {i+1}')
plt.tight_layout()
# Salida esperada: grid 2x2 de series aleatorias con ejes compartidos
```

**Parámetros clave de `subplots()`**: `figsize`, `sharex` / `sharey` (compartir ejes entre subplots), `gridspec_kw` (control fino de proporciones), `squeeze` (controlar si axes es array 2D o escalar).

## 3. Personalización: spines, ticks, labels, grids

Los spines son los bordes del Axes. Por defecto matplotlib muestra 4 (top, bottom, left, right). Lo más común es ocultar top y right.

```python
fig, ax = plt.subplots(figsize=(6, 3))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')

# Ocultar spines innecesarios
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Personalizar ticks
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax.tick_params(axis='both', labelsize=9)

# Grid sutil
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_xlabel('Ángulo (rad)'); ax.set_ylabel('sin(x)')
ax.legend(frameon=False)
# Salida esperada: curva sinusoidal con spines minimalistas, ticks LaTeX-like, grid punteado
```

Revisa [[Visualization Fundamentals]] para principios de data-ink ratio que aplican aquí.

## 4. Colores y estilos

matplotlib ofrece colormaps, estilos de línea y marcadores. El sistema `cycler` permite iterar colores automáticamente.

```python
fig, ax = plt.subplots(figsize=(6, 4))

# Colormaps: 'viridis', 'plasma', 'Blues', 'RdBu'
cmap = plt.get_cmap('viridis')
x = np.linspace(0, 10, 50)
for i in range(5):
    y = np.sin(x + i * 0.5) * (5 - i)
    ax.plot(x, y, color=cmap(i/5), linestyle=['-', '--', '-.', ':', '-'][i],
            marker=['o', 's', '^', 'D', 'v'][i], markevery=10, label=f'Serie {i+1}')

ax.legend()
ax.set_title('Colormap viridis + estilos de línea y marcadores')
# Salida esperada: 5 curvas con degradado viridis, estilos variados
```

**Linestyles**: `'-'`, `'--'`, `'-.'`, `':'`, `'None'`.
**Markers**: `'o'` (círculo), `'s'` (cuadrado), `'^'` (triángulo), `'D'` (diamante), `'.'` (punto).
**Cycler**: cambia el Color cycler con `plt.rc('axes', prop_cycle=plt.cycler(color=['#4C72B0', '#DD8452', '#55A868']))`.

## 5. Subplots avanzados: GridSpec, shared axes, inset axes

Cuando `subplots()` no basta, `GridSpec` da control absoluto sobre el layout.

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

ax_main = fig.add_subplot(gs[1, 1])       # Centro grande
ax_top = fig.add_subplot(gs[0, 1])        # Superior compartido
ax_left = fig.add_subplot(gs[1, 0])       # Izquierdo
ax_colorbar = fig.add_subplot(gs[2, 2])   # Esquina inferior derecha

# Datos de ejemplo
np.random.seed(0)
data = np.random.randn(20, 20)
ax_main.imshow(data, cmap='viridis', aspect='auto')
ax_top.plot(data.mean(axis=0))
ax_left.plot(data.mean(axis=1), range(20))
ax_colorbar.text(0.5, 0.5, 'Info extra', ha='center', va='center')

# Inset axes: zoom dentro del plot principal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
inset = inset_axes(ax_main, width='30%', height='30%', loc='upper right')
inset.hist(data.ravel(), bins=15, color='#4C72B0', edgecolor='white')
inset.set_title('Histograma', fontsize=8)

plt.tight_layout()
# Salida esperada: layout asimétrico con inset axes tipo zoom + histograma
```

**Shared axes**: `sharex=True` en subplots sincroniza zoom/pan. [[Interactive Visualization (plotly-altair)]] ofrece zoom interactivo nativo.

## 6. Anotaciones: text, arrows, highlights

Las anotaciones convierten un gráfico genérico en una comunicación precisa. Usa `ax.annotate()` para flechas + texto.

```python
fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 4*np.pi, 200)
y = np.sin(x)
ax.plot(x, y, color='#4C72B0')

# Punto destacado
x_max = 3*np.pi/2
y_max = np.sin(x_max)
ax.scatter([x_max], [y_max], color='#C44E52', s=100, zorder=5)

# Anotación con flecha
ax.annotate('Mínimo local', xy=(x_max, y_max), xytext=(x_max - 1, y_max - 0.5),
            arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# Ecuación LaTeX
ax.text(0.5, 0.95, r'$y = \sin(x)$', transform=ax.transAxes,
        fontsize=14, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_title('Anotaciones con texto, flechas y LaTeX')
# Salida esperada: sinusoidal con punto rojo, flecha apuntando al mínimo, ecuación LaTeX
```

**Expresiones matemáticas**: encierra en `$...$` (LaTeX inline) o `$$...$$` (display). Usa `r''` raw strings para evitar escapes. Ejemplo: `r'$\alpha > \beta^2$'`.

## 7. Guardar figuras: dpi, formatos, transparent background

`fig.savefig()` tiene opciones que marcan la diferencia entre un gráfico "que se ve bien" y uno profesional.

```python
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([0, 1, 2], [0, 1, 4], linewidth=3)
ax.set_title('Exportación profesional')

# Opciones clave de savefig
fig.savefig('figura.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
# Salida esperada: archivo PNG de alta resolución, recortado al contenido

# Para papers (vectorial + transparencia)
fig.savefig('figura.pdf', bbox_inches='tight', transparent=True)
# Salida esperada: PDF con fondo transparente, ideal para overlays en LaTeX

# Para web (SVG editable)
fig.savefig('figura.svg', bbox_inches='tight')
# Salida esperada: SVG que puede editarse en Illustrator o Inkscape
```

**Parámetros**: `dpi=300` (mínimo para print), `bbox_inches='tight'` (recorta whitespace), `transparent=True` (sin fondo), `format='pdf'|'png'|'svg'|'eps'`. [[CLI & Productivity]] automatiza la exportación batch de figuras.

## 8. Common Mistakes

| Error | Síntoma | Solución |
|---|---|---|
| Crear 50 figuras en loop | Memory leak, kernel crash | `plt.close(fig)` o reutilizar axes |
| No llamar `plt.close()` | Acumulación en RAM | Siempre cerrar figuras no usadas |
| Modificar `rcParams` sin respaldo | Cambios globales afectan otros plots | Usar `plt.style.context('ggplot')` |
| Usar `plt.` vs `ax.` mezclado | Confusión en código largo | Preferir `fig, ax = plt.subplots()` |
| No llamar `tight_layout()` | Ejes solapados, títulos cortados | Siempre `plt.tight_layout()` al final |

```python
# ❌ Mal: crear figuras nuevas en cada iteración
for i in range(10):
    fig, ax = plt.subplots()
    ax.plot(np.random.randn(10))
    # fig no se cierra → memory leak

# ✅ Bien: crear y cerrar
for i in range(10):
    fig, ax = plt.subplots()
    ax.plot(np.random.randn(10))
    plt.close(fig)  # Libera memoria
```

## Resumen

1. Figure es el canvas contenedor; Axes es donde se dibujan los datos. Siempre usa `fig, ax = plt.subplots()`.
2. `plt.subplots()` crea Figure + N Axes en una línea; usa `sharex`/`sharey` para ejes sincronizados.
3. Personaliza spines (oculta top/right), ticks (frecuencia y etiquetas), y grids (sutiles).
4. Elige colormaps según el tipo de variable: categórica (Set2), secuencial (viridis), divergente (RdBu).
5. GridSpec permite layouts asimétricos; inset_axes crea zooms dentro del plot principal.
6. Las anotaciones (flechas + LaTeX) convierten un gráfico genérico en comunicación precisa.
7. Guarda en PNG (300 dpi), PDF (transparente, vectorial), o SVG (editable en Illustrator).
8. Siempre cierra figuras con `plt.close(fig)` en loops para evitar memory leaks.

## Check Your Understanding

1. ¿Cuál es la diferencia entre Figure y Axes?
   *Respuesta: Figure es el canvas contenedor; Axes es el área donde se dibujan los datos. Una Figure puede tener múltiples Axes.*

2. ¿Qué hace `bbox_inches='tight'` en `savefig()`?
   *Respuesta: Recorta el whitespace alrededor del gráfico para que la figura exportada tenga márgenes mínimos.*

3. ¿Por qué debes evitar mezclar `plt.plot()` y `ax.plot()` en el mismo script?
   *Respuesta: `plt.plot()` opera sobre el "axes actual" global, lo que puede causar comportamiento inesperado si tienes múltiples subplots. `ax.plot()` es explícito y seguro.*

4. ¿Qué problema resuelve `tight_layout()`?
   *Respuesta: Ajusta automáticamente el espaciado entre subplots para evitar solapamiento de etiquetas, títulos y ticks.*

5. ¿Cómo evitas un memory leak al generar 100 figuras en un loop?
   *Respuesta: Llamando `plt.close(fig)` al final de cada iteración para liberar la memoria de la figura.*

## Where to Go Next

- [[Visualization Fundamentals]]
- [[seaborn & Statistical Plots]]
- [[Interactive Visualization (plotly-altair)]]
- [[Python for Data Science]]
- [[CLI & Productivity]]
