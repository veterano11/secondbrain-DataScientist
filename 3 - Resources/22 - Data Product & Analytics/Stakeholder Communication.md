---
tags:
  - communication
  - stakeholder
  - presentation
  - data-storytelling
  - leadership
  - soft-skills
status: seedling
created: 2026-06-28
---

# Stakeholder Communication

## 1. Escenario de aprendizaje

Pasaste tres días investigando por qué las ventas cayeron 15% en el último trimestre. Encontraste la causa: un cambio en la estructura de precios que alienó al segmento de usuarios free → premium. Tu análisis es impecable, los datos están limpios, las visualizaciones son correctas.

Llegás a la reunión con la C-suite. Mostrás tu primer slide: un diagrama de dispersión con 4 variables, colores, tamaños, y una línea de regresión. Silencio. El CFO pregunta: "¿y esto qué significa?" El CEO mira el reloj.

La comunicación con stakeholders no es "dumb down" tu trabajo. Es **traducción**: convertir complejidad analítica en claridad ejecutiva. Este note te da frameworks para lograrlo.

## 2. Requisitos

- Experiencia haciendo análisis de datos (no es un note introductorio)
- Familiaridad con dashboards y visualizaciones
- Haber presentado a audiencias no técnicas al menos una vez

## 3. Tres reglas de oro

Antes de cualquier presentación, aplicá estas 3 reglas:

### Regla 1: Conocé a tu audiencia
No es lo mismo presentar al equipo de producto que al board. Preguntate:
- ¿Cuánto tiempo tienen? (15 min vs 1 hora)
- ¿Qué nivel de detalle necesitan? (KPI summary vs drill-down)
- ¿Qué decisión necesitan tomar después de escucharte?

| Audiencia | Tiempo | Formato | Decisión típica |
|-----------|--------|---------|-----------------|
| C-suite | 10-15 min | 3 slides + oral | Aprobar/rechazar inversión |
| Equipo de producto | 30 min | Workshop con datos | Priorizar backlog |
| Ingeniería | 45 min | Technical deep-dive | Cambiar implementación |

### Regla 2: Empezá por la conclusión
Nunca empieces con "primero veamos la metodología". Empezá con:

> "Las ventas cayeron 15% por un cambio en precios que afectó a usuarios con más de 6 meses de antigüedad. Recomiendo revertir el cambio para el segmento afectado."

Después mostrás los datos que sustentan esa conclusión.

### Regla 3: Una idea por slide
Si un slide necesita más de una idea, partilo. La carga cognitiva de interpretar un gráfico ya es alta. Si encima mezclás dos insights, perdés a tu audiencia.

## 4. La estructura Minto (Pyramid Principle)

Barbara Minto, consultora de McKinsey, desarrolló el **Pyramid Principle**: estructura tu comunicación como una pirámide donde la conclusión está arriba y los argumentos la soportan.

```
                        Conclusión
                        /    |    \
                   Argumento A  B   C
                    /    \      |    / \
                 Dato  Dato  Dato  D Dato
```

Para nuestro escenario:

1. **Respuesta (conclusión)**: El cambio de precios de mayo causó la caída de ventas en usuarios longevos.
2. **Contexto**: En mayo lanzamos nueva estructura de precios. Las ventas venían estables.
3. **Conflicto**: 3 semanas después, las ventas del segmento 6+ meses cayeron 25%.
4. **Pregunta**: ¿Fue el precio la causa y qué hacemos?

```sql
-- Query para sustentar el argumento: ventas pre/post cambio de precios
WITH pre_change AS (
    SELECT
        u.user_segment,
        COUNT(DISTINCT s.subscription_id) AS subscriptions,
        SUM(s.amount) AS revenue
    FROM subscriptions s
    JOIN users u ON s.user_id = u.user_id
    WHERE s.subscription_date BETWEEN '2026-04-01' AND '2026-04-30'
    GROUP BY u.user_segment
),
post_change AS (
    SELECT
        u.user_segment,
        COUNT(DISTINCT s.subscription_id) AS subscriptions,
        SUM(s.amount) AS revenue
    FROM subscriptions s
    JOIN users u ON s.user_id = u.user_id
    WHERE s.subscription_date BETWEEN '2026-05-15' AND '2026-06-15'
    GROUP BY u.user_segment
)
SELECT
    COALESCE(pre.user_segment, post.user_segment) AS segment,
    pre.subscriptions AS pre_subs,
    pre.revenue AS pre_revenue,
    post.subscriptions AS post_subs,
    post.revenue AS post_revenue,
    ROUND(100.0 * (post.revenue - pre.revenue) / NULLIF(pre.revenue, 0), 2) AS revenue_change_pct
FROM pre_change pre
FULL OUTER JOIN post_change post ON pre.user_segment = post.user_segment
ORDER BY revenue_change_pct;
```

**Salida esperada:**
```
segment       | pre_subs | pre_revenue | post_subs | post_revenue | revenue_change_pct
--------------|---------|------------|----------|-------------|-------------------
0-3 months    | 1200    | 60000      | 1250     | 62500       | 4.17
3-6 months    | 800     | 40000      | 780      | 39000       | -2.5
6+ months     | 1500    | 75000      | 1125     | 56250       | -25.0
```

El segmento 6+ meses cayó 25%. Ese es tu gancho narrativo.

## 5. Data Storytelling: ancla, insight, acción

Cada gráfico o tabla que mostrés debe tener tres capas:

| Capa | Pregunta | Ejemplo |
|------|----------|---------|
| **Ancla** | ¿Qué estoy mirando? | "Este gráfico muestra revenue por segmento de usuario antes y después del cambio de precios." |
| **Insight** | ¿Qué revela? | "El segmento 6+ meses cayó 25% mientras los demás se mantuvieron estables." |
| **Acción** | ¿Qué hacemos? | "Propongo revertir el cambio de precios solo para usuarios con más de 6 meses de antigüedad." |

```python
import matplotlib.pyplot as plt
import pandas as pd

# Datos del query anterior
data = pd.DataFrame({
    'segment': ['0-3 months', '3-6 months', '6+ months'],
    'pre_revenue': [60000, 40000, 75000],
    'post_revenue': [62500, 39000, 56250]
})

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(data))
width = 0.35

ax.bar([i - width/2 for i in x], data['pre_revenue'], width, label='Pre (April)', color='#2ecc71')
ax.bar([i + width/2 for i in x], data['post_revenue'], width, label='Post (May-Jun)', color='#e74c3c')

ax.set_xticks(x)
ax.set_xticklabels(data['segment'])
ax.set_ylabel('Revenue ($)')
ax.set_title('Revenue Change by User Segment')
ax.legend()

# Anotar el insight clave
ax.annotate('-25%', xy=(2, 56250), xytext=(2, 30000),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=14, fontweight='bold', ha='center')

plt.tight_layout()
plt.show()
```

**Salida esperada:**
```
Gráfico de barras con dos barras por segmento (pre y post).
Una flecha roja apunta a la caída del segmento 6+ meses con el texto "-25%".
```

No mostrés el código en la presentación. Mostrá el gráfico final con el insight anotado.

## 6. Dashboard design

Un dashboard no es un dump de datos. Es una **jerarquía visual** que guía al ojo:

```
┌──────────────────────────────────────────────────┐
│  KPI 1: $1.2M (+4%) │ KPI 2: 15K (-25%) │ ...   │  ← Norte: KPIs críticos
├──────────────────────────────────────────────────┤
│                                                  │
│  Gráfico principal: Revenue over time            │  ← Centro: tendencia
│                                                  │
├────────────────────┬─────────────────────────────┤
│ Segment breakdown  │ Table with drill-down       │  ← Base: detalle
│ por cohort         │ filters                      │
└────────────────────┴─────────────────────────────┘
```

Reglas de dashboard:

- **KPI arriba siempre**: los números más importantes en la primera fila
- **Mínimo contexto necesario**: título, fecha, filtro de segmento
- **Consistencia de color**: rojo = negativo, verde = positivo (o azul si hay daltonismo en tu audiencia)
- **Filtros lógicos**: fecha, segmento, fuente — no más de 3 filtros visibles

```sql
-- Query para el KPI principal del dashboard
WITH revenue_kpi AS (
    SELECT
        DATE_TRUNC('month', subscription_date) AS month,
        SUM(amount) AS mrr,
        COUNT(DISTINCT user_id) AS paying_users,
        ROUND(SUM(amount) / NULLIF(COUNT(DISTINCT user_id), 0), 2) AS arpu
    FROM subscriptions
    WHERE subscription_date >= '2026-01-01'
    GROUP BY 1
)
SELECT
    month,
    mrr,
    LAG(mrr) OVER (ORDER BY month) AS prev_month_mrr,
    ROUND(100.0 * (mrr - LAG(mrr) OVER (ORDER BY month)) / NULLIF(LAG(mrr) OVER (ORDER BY month), 0), 2) AS mom_change,
    paying_users,
    arpu
FROM revenue_kpi
ORDER BY month DESC;
```

**Salida esperada:**
```
month      | mrr     | prev_month_mrr | mom_change | paying_users | arpu
-----------|---------|---------------|-----------|-------------|------
2026-06-01 | 1200000 | 1250000       | -4.0      | 15000       | 80.0
2026-05-01 | 1250000 | 1280000       | -2.34     | 16500       | 75.76
```

## 7. Presentación oral: manejo de objeciones

La presentación oral es donde los análisis mueren o viven. Estrategias:

- **"So what?" test**: después de cada afirmación, preguntate "¿y qué implica esto para la decisión que tiene que tomar mi audiencia?"
- **Executive summary de 30 segundos**: si solo tenés 30 segundos, ¿qué decís? Practicalo.
- **Anticipar objeciones**: antes de la reunión, pensá en las 3 objeciones más probables y prepará datos para responderlas.

| Objeción típica | Preparación |
|----------------|-------------|
| "¿Y cómo sabés que fue el precio y no otra cosa?" | Mostrá que otros segmentos no cambiaron (grupo de control). |
| "Pero el revenue total subió." | Segmentá: el total puede subir mientras un segmento clave cae. |
| "¿Qué tan seguro estás de los datos?" | Mostrá intervalo de confianza y volumen de datos. |

```sql
-- Preparando respuesta a objeción: ¿fue el precio o estacionalidad?
SELECT
    segment,
    AVG(revenue) AS avg_revenue,
    STDDEV(revenue) AS std_revenue,
    COUNT(*) AS months_observed,
    MIN(revenue) AS min_revenue,
    MAX(revenue) AS max_revenue
FROM monthly_segment_revenue
WHERE segment = '6+ months'
  AND month BETWEEN '2025-01-01' AND '2026-03-31'  -- 15 meses pre-cambio
GROUP BY segment;
```

**Salida esperada:**
```
segment   | avg_revenue | std_revenue | months_observed | min_revenue | max_revenue
----------|------------|------------|----------------|------------|-----------
6+ months | 74500      | 2100       | 15             | 71000      | 78000
```

La desviación estándar pre-cambio es $2.100. La caída post-cambio fue $18.750 (de $75.000 a $56.250). Son casi 9 desviaciones estándar. No es estacionalidad — es un quiebre estructural.

## Resumen

- **Conocé a tu audiencia**: adaptá formato, tiempo y nivel de detalle.
- **Empezá por la conclusión**: el Pyramid Principle de Minto.
- **Data storytelling**: cada gráfico necesita ancla, insight y acción.
- **Dashboard design**: jerarquía visual con KPI arriba y detalle abajo.
- **Manejo de objeciones**: anticipá preguntas y prepará datos de respaldo.
- **Una idea por slide**: reducir carga cognitiva de la audiencia.
- **Executive summary de 30 segundos**: si no podés resumirlo, no lo entendiste.

## Common Mistakes

- **Mostrar todo el análisis**: la C-suite no necesita ver los 20 queries que corriste. Mostrá solo los resultados relevantes.
- **No tener recomendación**: "los datos muestran que las ventas cayeron" no es útil. "Recomiendo revertir el cambio de precios para el segmento 6+ meses" es útil.
- **Jerga técnica**: "p-value", "heterocedasticidad", "endogeneidad" no existen en una reunión de Negocios. Usá lenguaje coloquial.
- **Visualización engañosa**: ejes truncados, colores confusos, 3D innecesario. Cada elección visual debe hacer el insight más claro, no más confuso.
- **No practicar**: la primera vez que escuchás tu presentación no debería ser en la reunión.
- **No leer la sala**: si ves que el CEO mira el celular, saltá al final. Tené un "skip to conclusions" preparado.

## Check Your Understanding

1. El equipo de producto te pide que presentes un análisis de abandono de 30 minutos. El CEO se suma. ¿Qué hacés? <!-- Prepará dos versiones: una de 30 min con detalle para producto y un executive summary de 5 min para el CEO. Preguntá al inicio cuánto tiempo tienen realmente. -->
2. Convertí este hallazgo técnico a una frase para la C-suite: "El odds ratio de churn para usuarios con sesiones < 2 por semana es 3.2 (IC 95%: 2.8-3.6, p < 0.001) comparado con usuarios con sesiones >= 2." <!-- "Los usuarios que entrenan menos de 2 veces por semana abandonan 3 veces más que los que entrenan más." -->
3. Un stakeholder dice: "No entiendo este gráfico." ¿Respondés explicando el gráfico o mostrando otro? <!-- Mostrá otro. Si no se entiende a la primera, rediseñalo. La explicación verbal no compensa un mal diseño visual. -->
4. ¿Por qué el Pyramid Principle funciona para la C-suite pero no necesariamente para un equipo técnico? <!-- La C-suite necesita decisión rápida (conclusión primero). El equipo técnico necesita entender el proceso para validar y ejecutar (contexto primero). Una misma estructura no sirve para todas las audiencias. -->
5. Tu dashboard muestra 20 KPIs en una página. ¿Cuál es el problema? <!-- La audiencia no sabe qué mirar. Un dashboard debe priorizar 3-5 KPIs críticos y esconder el detalle detrás de filtros o drill-downs. Si todo es importante, nada es importante. -->

## Where to Go Next

- [[Storytelling with Data]] — el libro de Cole Nussbaumer Knaflic en formato note.
- [[Visualization Fundamentals]] — cómo elegir el gráfico correcto para cada mensaje.
- [[Data Product Thinking]] — conectá la comunicación con la north star metric que la C-suite ya conoce.
- [[Self-Serve Analytics]] — cuantos más stakeholders puedan responderse solos, menos reuniones de "decime qué dice este número".
- [[Experimentation & Growth]] — cuando presentes resultados de experimentos, esta nota te da la estructura.
- [[CLI & Productivity]] — herramientas para generar presentaciones desde datos automatizadas.
