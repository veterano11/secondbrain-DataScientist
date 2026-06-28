---
tags:
  - plotly
  - altair
  - interactive
  - visualization
  - dashboard
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu dashboard de ventas en PDF estático ya no sirve. El equipo de producto quiere filtrar por región, hacer zoom en fechas, y ver detalles al pasar el mouse. Necesitas visualizaciones interactivas. Esta nota cubre Plotly Express y Altair para construir dashboards que la gente *quiera* usar.

## 1. ¿Por qué interactivo?

Los gráficos estáticos muestran una historia fija. Los interactivos permiten exploración libre. Casos de uso clave:

- **Zoom & Pan**: examinar regiones densas (ej. ventas diarias por 2 años).
- **Tooltips**: mostrar métricas exactas al hacer hover, sin saturar el gráfico.
- **Animation**: mostrar evolución temporal (animación de burbujas tipo Gapminder).
- **Filtering**: seleccionar categorías en un gráfico y reflejar el cambio en otros.

```python
# --- zoom, tooltips, animation, filtering ---
import plotly.express as px

df = px.data.gapminder()
fig = px.scatter(
    df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
    hover_name="country", log_x=True, size_max=60,
    animation_frame="year", range_x=[200, 60000], range_y=[25, 90],
    title="Gapminder: PIB vs Esperanza de vida (1952–2007)"
)
fig.show()
# Salida esperada: scatter plot animado con burbujas que crecen/avanzan por año.
```

## 2. Plotly Express: alto nivel, resultados rápidos

`plotly.express` (px) genera figuras completas con una línea. Soportado: `scatter`, `line`, `bar`, `histogram`, `box`, `violin`, `density_heatmap`, `choropleth`, `sunburst`.

```python
import plotly.express as px

df = px.data.iris()
fig = px.scatter(
    df, x="sepal_width", y="sepal_length",
    color="species", size="petal_length",
    hover_data=["petal_width"],
    facet_col="species",
    title="Iris — dimensiones por especie"
)
fig.show()
# Salida esperada: 3 facetas (una por especie), puntos coloreados y con tooltip.
```

**Parámetros clave:** `color`, `size`, `symbol`, `facet_col`, `facet_row`, `hover_data`, `animation_frame`, `log_x`, `range_x`, `trendline`.

## 3. Plotly Figure: layout, updatemenus, sliders

Toda figura Plotly es un diccionario JSON anidado. Se personaliza con `fig.update_layout()` y `fig.update_traces()`. Puedes agregar controles interactivos.

```python
import plotly.graph_objects as go
import numpy as np

x = np.linspace(0, 10, 100)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=np.sin(x), mode="lines", name="sin"))
fig.add_trace(go.Scatter(x=x, y=np.cos(x), mode="lines", name="cos"))

fig.update_layout(
    title="Funciones trigonométricas",
    xaxis_title="x", yaxis_title="f(x)",
    hovermode="x unified",
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(label="Mostrar ambas", method="update",
                     args=[{"visible": [True, True]}]),
                dict(label="Solo sin", method="update",
                     args=[{"visible": [True, False]}]),
                dict(label="Solo cos", method="update",
                     args=[{"visible": [False, True]}]),
            ],
        )
    ],
)
fig.show()
# Salida esperada: gráfico de líneas con botones para ocultar/mostrar cada trazo.
```

```python
# --- Slider para umbral ---
fig = go.Figure()
for i, freq in enumerate([1, 2, 3, 4]):
    fig.add_trace(go.Scatter(x=x, y=np.sin(freq * x), mode="lines",
                             visible=(i == 0), name=f"freq={freq}"))

fig.update_layout(
    title="Frecuencia variable",
    sliders=[dict(
        steps=[dict(method="update", args=[{"visible": [i == j for j in range(4)]}],
                    label=f"{f}Hz") for i, f in enumerate([1, 2, 3, 4])]
    )]
)
fig.show()
# Salida esperada: slider que cambia la frecuencia de la onda sinusoidal.
```

## 4. Altair: gramática declarativa

Altair sigue la gramática de gráficos de Wilkinson. Separas datos (`Chart`) de canales visuales (`mark_*`) y codificaciones (`encode`). Cada canal es una columna del DataFrame.

```python
import altair as alt
from vega_datasets import data

df = data.cars()
alt.Chart(df).mark_circle().encode(
    x="Horsepower:Q",
    y="Miles_per_Gallon:Q",
    color="Origin:N",
    size="Acceleration:Q",
    tooltip=["Name:N", "Year:O"]
).properties(
    title="Autos — HP vs MPG por origen"
)
# Salida esperada: scatter plot interactivo con tooltips, colores y tamaños.
```

**Canales principales:** `x`, `y`, `color`, `size`, `shape`, `opacity`, `row`, `column`, `tooltip`. Tipos: `:Q` (cuantitativo), `:N` (nominal), `:O` (ordinal), `:T` (temporal).

## 5. Altair interactividad: selections y condiciones

Altair permite `selection_interval()` (zoom/brush) y `selection_multi()` (clic en puntos). Se combinan con `condition()` para cambiar estilo.

```python
import altair as alt
from vega_datasets import data

df = data.cars()

selection = alt.selection_interval()

alt.Chart(df).mark_circle().encode(
    x="Horsepower:Q",
    y="Miles_per_Gallon:Q",
    color=alt.condition(selection, "Origin:N", alt.value("lightgray"))
).add_selection(
    selection
).properties(
    title="Selecciona un área para resaltar"
)
# Salida esperada: scatter donde los puntos fuera del brush se vuelven grises.
```

```python
# --- Multi-selection con binding y filter ---
from vega_datasets import data

df = data.cars()
selector = alt.selection_multi(fields=["Origin"])

base = alt.Chart(df).mark_circle().encode(
    x="Horsepower:Q", y="Miles_per_Gallon:Q",
    color=alt.condition(selector, "Origin:N", alt.value("lightgray")),
    size="Acceleration:Q"
).add_selection(selector)

alt.hconcat(
    base,
    base.encode(color="Origin:N").add_selection(selector)
).properties(title="Clic en una categoría para filtrar")
# Salida esperada: dos paneles, clic en "Europe" resalta puntos de Europa en ambos.
```

## 6. Dashboards: Plotly Dash vs Streamlit

Para dashboards completos usas **Dash** (Plotly ecosistema) o **Streamlit** (general purpose).

| Aspecto | Dash | Streamlit |
|---------|------|-----------|
| Lenguaje | Python + HTML/CSS | Solo Python |
| Reactividad | Callbacks explícitos | Re-run script completo |
| Curva | Media-Alta | Baja |
| Layout | `html.Div`, `dcc.Graph` | `st.columns`, `st.sidebar` |
| Ideal | Dashboards complejos y custom | Prototipos rápidos y reportes |

```python
# --- Mini app Dash (conceptual) ---
# from dash import Dash, dcc, html, Input, Output
# import plotly.express as px
#
# app = Dash(__name__)
# app.layout = html.Div([
#     dcc.Dropdown(id="region", options=[...]),
#     dcc.Graph(id="sales-chart")
# ])
#
# @app.callback(
#     Output("sales-chart", "figure"),
#     Input("region", "value")
# )
# def update_chart(region):
#     df = load_data(region)
#     return px.line(df, x="date", y="sales")
#
# if __name__ == "__main__":
#     app.run(debug=True)
# Salida esperada: servidor web con un dropdown que actualiza el gráfico.
```

## 7. Common Mistakes

- **Demasiada interactividad**: 10 sliders + 3 dropdowns + brush abruma al usuario. Prioriza 1–2 controles principales.
- **Sin estados de carga**: Plotly procesa del lado del cliente, pero si filtras 500k puntos, el navegador se congela. Considera agregaciones previas.
- **Tooltips sin formato**: números con 10 decimales. Usa `:,.2f` o `alt.Tooltip(field="sales", format=",.0f")`.
- **Ignorar la audiencia**: un científico de datos quiere granularidad; un ejecutivo quiere la tendencia. Diseña para el usuario final.
- **Falta de contexto**: un gráfico interactivo sin título, ejes rotulados ni anotaciones no es auto-contenido. La interactividad complementa, no reemplaza, el diseño.
- **Plotly figuras lentas con muchos puntos**: usa `plotly.graph_objects.Scattergl` para WebGL o agrega `partial` con `sample()`.
- **Altair limita a 5000 filas**: cambia `alt.data_transformers.enable("default")` o usa `vega_datasets` para manejo eficiente.

```python
# --- Buenas prácticas: tooltips formateados ---
import plotly.express as px
df = px.data.gapminder().query("year == 2007")
fig = px.scatter(
    df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
    hover_data={"gdpPercap": ":,.0f", "pop": ":,.0f"},
    title="Tooltips formateados correctamente"
)
fig.show()
# Salida esperada: tooltip con GDP en miles separados por coma.
```

## Resumen

1. Plotly Express genera gráficos interactivos con una línea; ideal para exploración rápida.
2. Plotly Figure (`go.Figure`) permite controles avanzados: botones, sliders, updatemenus.
3. Altair usa gramática declarativa con canales (`x`, `y`, `color`) y selections para interactividad nativa.
4. La interactividad debe ser intencional: cada control debe responder a una pregunta del usuario.
5. Dash es ideal para dashboards complejos con callbacks; Streamlit para prototipos rápidos.
6. El rendimiento importa: agrega datos antes de graficar y formatea tooltips.
7. La interactividad no excusa mal diseño: título, ejes, contexto y anotaciones siguen siendo necesarios.

## Check Your Understanding

1. ¿Qué diferencia a Plotly Express de Plotly Figure?  
   *Express es alto nivel (una línea = gráfico completo); Figure es bajo nivel (control total sobre trazos y layout).*

2. ¿Cómo harías un scatter plot en Altair donde al hacer clic en una categoría se resalten esos puntos?  
   *Con `selection_multi(fields=["categoria"])` y `color=alt.condition(selection, "categoria:N", alt.value("lightgray"))`.*

3. ¿Cuándo conviene usar Dash en lugar de Streamlit?  
   *Cuando necesitas callbacks complejos, layout custom con HTML/CSS, o dashboards con múltiples fuentes de datos reactivas.*

4. ¿Qué problema tiene Altair con datasets > 5000 filas y cómo se soluciona?  
   *Altair incrusta datos en el JSON; se soluciona con `alt.data_transformers.enable("vegajs")` o agregando los datos.*

5. ¿Por qué demasiados controles interactivos pueden ser contraproducentes?  
   *Parálisis de decisión: el usuario no sabe por dónde empezar. Cada control debe responder una pregunta específica.*

## Where to Go Next

- [[seaborn & Statistical Plots]] — gráficos estadísticos estáticos, complemento a Plotly.
- [[matplotlib en profundidad]] — control total sobre gráficos, base de Plotly.
- [[Visualization Fundamentals]] — principios generales de visualización.
- [[Storytelling with Data]] — cómo estructurar una narrativa visual persuasiva.
- [[Python for Data Science]] — ecosistema Python para análisis y visualización.
