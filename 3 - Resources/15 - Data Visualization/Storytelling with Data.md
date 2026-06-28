---
tags:
  - storytelling
  - narrative
  - presentation
  - communication
  - dashboard
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Los datos muestran que las ventas bajaron 15% este trimestre. Tu jefe quiere saber por qué en 5 minutos. No puedes lanzarle 20 gráficos. Necesitas una narrativa visual clara que cuente la historia. Esta nota cubre cómo estructurar y presentar datos de forma persuasiva ante la C-suite.

## 1. El arco narrativo: contexto → conflicto → resolución

Toda buena historia de datos sigue tres actos. No presentes los hallazgos en orden cronológico de análisis; preséntalos en orden narrativo.

| Acto | Pregunta | Ejemplo ventas |
|------|----------|----------------|
| **Contexto** | ¿Qué esperábamos? | Crecimiento proyectado: +8% vs Q2 del año pasado |
| **Conflicto** | ¿Qué pasó? | Caída real: -15%. Brecha de 23 puntos. |
| **Resolución** | ¿Qué hacemos? | Canal B2B cayó 40% (cambio de pricing). Recomendación: revertir precio o aumentar descuentos por volumen. |

```python
# --- Visualización del arco narrativo ---
import matplotlib.pyplot as plt
import numpy as np

periodos = ["Q1", "Q2 (proyectado)", "Q2 (real)"]
valores = [120, 130, 102]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(periodos, valores, color=["#3498db", "#2ecc71", "#e74c3c"])
ax.axhline(y=120, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Ventas (M USD)")
ax.set_title("Contexto → Conflicto: la brecha de 23 puntos")

for bar, val in zip(bars, valores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"${val}M", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()
# Salida esperada: barras verde (proyectado) y roja (real) con la brecha visible.
```

## 2. La regla del headline

El título de tu gráfico debe ser la conclusión, no la descripción. "Ventas Q2 2026" no ayuda. "Ventas cayeron 15% por cambio de pricing en B2B" es accionable.

```python
# --- Mal título vs buen título ---
import matplotlib.pyplot as plt

meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"]
ventas = [105, 110, 108, 95, 88, 82]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Mal título
ax1.plot(meses, ventas, marker="o")
ax1.set_title("Ventas 2026")
ax1.set_ylabel("M USD")

# Buen título
ax2.plot(meses, ventas, marker="o", color="#e74c3c")
ax2.set_title("Ventas en caída libre: -22% desde marzo")
ax2.set_ylabel("M USD")
ax2.axvspan(2.5, 5.5, alpha=0.08, color="red")

plt.tight_layout()
plt.show()
# Salida esperada: dos gráficos iguales, pero el segundo comunica el mensaje al instante.
```

## 3. Decluttering: menos es más

Cada elemento visual debe tener un propósito. Elimina: líneas de grilla redundantes, bordes de gráfico, leyendas que se explican solas, etiquetas repetitivas, colores decorativos sin significado.

```python
# --- Antes y después de decluttering ---
import matplotlib.pyplot as plt
import numpy as np

categorias = ["B2B", "B2C", "Gov", "Other"]
valores = [40, 30, 20, 10]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Antes: sobrecargado
ax1.bar(categorias, valores, color=["red", "blue", "green", "purple"])
ax1.set_title("Distribución de Ventas por Canal")
ax1.set_xlabel("Canal")
ax1.set_ylabel("Porcentaje")
ax1.grid(True, axis="y", alpha=0.5)
ax1.spines["top"].set_visible(True)
ax1.spines["right"].set_visible(True)

# Después: limpio
colors_clean = ["#2c3e50", "#3498db", "#95a5a6", "#bdc3c7"]
ax2.bar(categorias, valores, color=colors_clean, width=0.5)
ax2.set_title("B2B domina: 40% del ingreso total", fontweight="bold")
ax2.set_ylabel("% del total")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#cccccc")
ax2.spines["bottom"].set_color("#cccccc")
ax2.tick_params(colors="#555555")
ax2.yaxis.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
# Salida esperada: izquierda desordenado, derecha minimalista y profesional.
```

## 4. Anotación estratégica

Guía la atención del lector. Usa texto, flechas, formas y colores para destacar el punto clave. No asumas que el lector "ya lo ve".

```python
# --- Anotaciones con flechas y texto ---
import matplotlib.pyplot as plt

meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"]
ventas = [105, 110, 108, 95, 70, 82]
eventos = {"Abr": "Nuevo pricing\nB2B", "May": "Caída cliente\nMayorista X"}

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(meses, ventas, marker="o", linewidth=2, color="#2c3e50")
ax.fill_between(range(len(meses)), ventas, alpha=0.08, color="#e74c3c")

for i, mes in enumerate(meses):
    if mes in eventos:
        ax.annotate(
            eventos[mes],
            xy=(i, ventas[i]),
            xytext=(i, ventas[i] + 12),
            ha="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#e74c3c", alpha=0.9)
        )

ax.set_title("Dos eventos explican la caída de mayo", fontweight="bold")
ax.set_ylabel("Ventas (M USD)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
# Salida esperada: línea descendente con anotaciones que conectan eventos a la caída.
```

## 5. Dashboards efectivos: jerarquía visual

Un dashboard no es una colección de gráficos. Es una narrativa con jerarquía: KPI más importante arriba a la izquierda, detalle abajo. Usa el patrón F de lectura.

```python
# --- Layout conceptual de dashboard ---
import matplotlib.pyplot as plt

fig, axes = plt.subplot_mosaic(
    [
        ["kpi1", "kpi2", "kpi3", "kpi4"],
        ["trend", "trend", "pie", "pie"],
        ["table", "table", "table", "table"],
    ],
    figsize=(12, 8),
    per_subplot_kw={"kpi1": {}, "kpi2": {}, "kpi3": {}, "kpi4": {},
                    "trend": {}, "pie": {}, "table": {}}
)

for key in ["kpi1", "kpi2", "kpi3", "kpi4"]:
    axes[key].text(0.5, 0.5, f"{key.upper()}\n$XXM", ha="center", va="center", fontsize=14)
    axes[key].set_frame_on(False)
    axes[key].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

axes["trend"].plot(["Ene", "Feb", "Mar", "Abr"], [100, 95, 88, 82], marker="o")
axes["trend"].set_title("Tendencia de Ventas")

axes["pie"].pie([40, 30, 20, 10], labels=["B2B", "B2C", "Gov", "Other"],
                autopct="%1.0f%%", startangle=90)

axes["table"].axis("off")
axes["table"].table(cellText=[["B2B", "$40M", "-40%"], ["B2C", "$30M", "+5%"],
                               ["Gov", "$20M", "+2%"], ["Other", "$10M", "-10%"]],
                    colLabels=["Canal", "Ventas", "Var %"],
                    cellLoc="center", loc="center")
axes["table"].set_title("Detalle por Canal", y=0.85)

fig.suptitle("Dashboard Ejecutivo — Ventas Q2 2026", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
# Salida esperada: dashboard 4 KPIs arriba, tendencia + pie en medio, tabla detalle abajo.
```

## 6. Dark patterns en visualización

Conocer las malas prácticas ayuda a evitarlas y a detectar manipulación en terceros.

| Práctica Engañosa | Ejemplo | Cómo evitarla |
|-------------------|---------|---------------|
| **Cherry-picking** | Mostrar solo el mes con peor desempeño sin contexto anual | Siempre muestra la serie completa o un periodo representativo |
| **Eje truncado** | Eje Y desde 80 en vez de 0, exagerando una caída del 5% | Empieza el eje en 0 (o márcalo claramente si no lo haces) |
| **Omisión de contexto** | "Despidos 10%" sin mencionar que la industria promedia 15% | Incluye benchmarks o datos de referencia |
| **Dual-axis engañoso** | Dos escalas Y diferentes que hacen ver correlación donde no la hay | Si usas dual axis, explícalo y etiqueta cada escala |
| **Escala no uniforme** | Intervalos temporales desiguales presentados como iguales | Usa escala lineal o marca las rupturas explícitamente |

```python
# --- Eje truncado vs escala honesta ---
import matplotlib.pyplot as plt

meses = ["Ene", "Feb", "Mar", "Abr"]
valores = [98, 97, 96, 95]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Engañoso: eje truncado
ax1.plot(meses, valores, marker="o", color="#e74c3c")
ax1.set_ylim(90, 100)
ax1.set_title("¡Caída dramática del 3%!", color="#e74c3c")

# Honesto: escala desde 0
ax2.plot(meses, valores, marker="o", color="#3498db")
ax2.set_ylim(0, 100)
ax2.set_title("Variación mínima del 3% sobre base 100")

plt.tight_layout()
plt.show()
# Salida esperada: dos gráficos, el de la izquierda exagera visualmente la caída vs el derecho.
```

## 7. Common Mistakes

- **Demasiado texto**: una diapositiva con 200 palabras y 3 gráficos no se lee. Cada gráfico debe entenderse en < 5 segundos.
- **Sin flujo narrativo**: mostrar todos los hallazgos sin orden. Usa el arco contexto → conflicto → resolución.
- **Ignorar a la audiencia**: la C-suite quiere recomendaciones, no código. Los analistas quieren metodología, no conclusiones sin respaldo.
- **Colores sin significado**: usar rojo/verde para variables no binarias, o no considerar daltonismo. Usa paletas como ColorBrewer o Viridis.
- **Ausencia de llamado a la acción**: el lector se queda pensando "¿y ahora qué?". Toda presentación de datos debe terminar con una decisión o próximo paso.
- **Sobre-anotación**: anotar cada punto satura. Anota solo lo excepcional (picos, valles, cambios de tendencia).
- **No iterar**: el primer dashboard nunca es el mejor. Pruébalo con alguien que no conozca los datos y observa dónde se pierde.

```python
# --- Checklist de storytelling (no ejecutable, solo conceptual) ---
checklist = """
☐ ¿El título del gráfico comunica la conclusión?
☐ ¿Eliminé gridlines, bordes y leyendas innecesarias?
☐ ¿Hay una anotación que guíe al hallazgo principal?
☐ ¿La audiencia puede entenderlo en < 5 segundos?
☐ ¿Termina con una recomendación o llamado a la acción?
☐ ¿Verifiqué que la escala no sea engañosa?
"""
print(checklist)
# Salida esperada: lista de verificación para revisar antes de presentar.
```

## Resumen

1. Toda historia de datos sigue un arco: contexto → conflicto → resolución.
2. El título del gráfico debe ser la conclusión, no la descripción.
3. Decluttering: elimina todo elemento visual sin propósito informativo.
4. Las anotaciones guían al lector hacia el hallazgo clave.
5. Los dashboards efectivos tienen jerarquía visual: KPI arriba, detalle abajo.
6. Conocer los dark patterns ayuda a diseñar de forma ética y detectar manipulación.
7. El storytelling con datos es iterativo: prueba con alguien ajeno al proyecto.
8. Sin un llamado a la acción, los datos no generan decisiones.

## Check Your Understanding

1. ¿Cuáles son los tres actos del arco narrativo aplicado a datos?  
   *Contexto (qué esperábamos) → Conflicto (qué pasó) → Resolución (qué hacemos).*

2. ¿Qué es la regla del headline y por qué importa?  
   *El título del gráfico debe ser la conclusión, porque la audiencia decide en segundos si presta atención.*

3. ¿Por qué truncar el eje Y puede considerarse un dark pattern?  
   *Exagera visualmente diferencias pequeñas, engañando al lector sobre la magnitud del cambio.*

4. ¿Qué criterio usarías para decidir qué anotar en un gráfico?  
   *Solo anotar lo excepcional: picos, valles, cambios de tendencia, puntos fuera del patrón esperado.*

5. ¿Cuál es la diferencia entre un dashboard y una colección de gráficos?  
   *Un dashboard tiene jerarquía visual, flujo narrativo y responde preguntas específicas; una colección solo muestra datos.*

## Where to Go Next

- [[Visualization Fundamentals]] — principios básicos antes de contar la historia.
- [[Interactive Visualization (plotly-altair)]] — cómo hacer que tu historia sea explorable.
- [[seaborn & Statistical Plots]] — gráficos estadísticos para respaldar tu narrativa.
- [[matplotlib en profundidad]] — control total sobre anotaciones y personalización.
- [[Model Evaluation]] — cómo comunicar resultados de modelos a audiencias no técnicas.
