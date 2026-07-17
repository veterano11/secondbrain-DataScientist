---
tags:
  - analytics
  - product-analytics
  - funnel
  - cohort
  - retention
  - sql
status: seedling
created: 2026-06-28
---

# Funnel & Cohort Analysis

## 1. Escenario de aprendizaje

Tu app de fitness tiene 100.000 descargas en los últimos 30 días, pero solo 5.000 usuarios activos semanales. Perdiste 95.000 personas en el camino. ¿En qué paso se fueron?

El CEO quiere respuestas: ¿es el registro muy largo? ¿el onboarding no engancha? ¿la primera experiencia de entrenamiento es confusa? Necesitás un funnel analysis para identificar el cuello de botella y un cohort analysis para saber si los cambios que ya hiciste están mejorando la retención.

Este note te enseña a construir ambos análisis en SQL, interpretar resultados y evitar las trampas clásicas de diagnóstico de producto.

## 2. Requisitos

- SQL avanzado: CTEs, window functions, agregaciones condicionales
- Familiaridad con eventos de producto (logging, event naming)
- Conceptos básicos de probabilidad condicional (conversión entre etapas)

## 3. Funnel Analysis: definir etapas

Un **funnel** representa los pasos secuenciales que un usuario debe completar para llegar a un objetivo. Cada etapa es un evento discreto y ordenado.

Para nuestra app de fitness las etapas son:

1. **Download** — descarga la app desde la App Store
2. **Signup** — crea cuenta (email, Google, Apple)
3. **Onboarding** — completa el flujo de configuración inicial (edad, peso, objetivos)
4. **First Workout** — completa su primer entrenamiento guiado
5. **Personalized Plan** — genera o recibe su plan personalizado

```sql
-- Funnel query con COUNT DISTINCT por etapa
WITH stages AS (
    SELECT
        user_id,
        MAX(CASE WHEN event = 'app_download'   THEN 1 ELSE 0 END) AS stage_download,
        MAX(CASE WHEN event = 'signup_complete' THEN 1 ELSE 0 END) AS stage_signup,
        MAX(CASE WHEN event = 'onboarding_done' THEN 1 ELSE 0 END) AS stage_onboarding,
        MAX(CASE WHEN event = 'first_workout'   THEN 1 ELSE 0 END) AS stage_first_workout,
        MAX(CASE WHEN event = 'plan_generated'  THEN 1 ELSE 0 END) AS stage_plan
    FROM events
    WHERE event_date BETWEEN '2026-06-01' AND '2026-06-30'
    GROUP BY user_id
)
SELECT
    'Download'           AS stage,
    COUNT(*)             AS users
FROM stages WHERE stage_download = 1
UNION ALL
SELECT
    'Signup',
    COUNT(*)
FROM stages WHERE stage_signup = 1
UNION ALL
SELECT
    'Onboarding',
    COUNT(*)
FROM stages WHERE stage_onboarding = 1
UNION ALL
SELECT
    'First Workout',
    COUNT(*)
FROM stages WHERE stage_first_workout = 1
UNION ALL
SELECT
    'Personalized Plan',
    COUNT(*)
FROM stages WHERE stage_plan = 1
ORDER BY users DESC;
```

**Salida esperada:**
```
stage             | users
------------------|-------
Download          | 100000
Signup            | 65000
Onboarding        | 40000
First Workout     | 15000
Personalized Plan | 5000
```

## 4. Drop-off: calcular conversión entre etapas

El funnel raw muestra números absolutos, pero el **drop-off** revela dónde se pierde la mayor proporción de usuarios.

```sql
-- Conversión etapa a etapa con drop-off rate
WITH funnel AS (
    SELECT 100000 AS stage_download, 65000 AS stage_signup,
           40000 AS stage_onboarding, 15000 AS stage_first_workout, 5000 AS stage_plan
)
SELECT
    'Download → Signup'             AS transition,
    stage_download                  AS entered,
    stage_signup                    AS converted,
    ROUND(100.0 * stage_signup / stage_download, 2) AS conversion_rate,
    ROUND(100.0 - (100.0 * stage_signup / stage_download), 2) AS drop_off
FROM funnel
UNION ALL
SELECT
    'Signup → Onboarding',
    stage_signup, stage_onboarding,
    ROUND(100.0 * stage_onboarding / NULLIF(stage_signup, 0), 2),
    ROUND(100.0 - (100.0 * stage_onboarding / NULLIF(stage_signup, 0)), 2)
FROM funnel
UNION ALL
SELECT
    'Onboarding → First Workout',
    stage_onboarding, stage_first_workout,
    ROUND(100.0 * stage_first_workout / NULLIF(stage_onboarding, 0), 2),
    ROUND(100.0 - (100.0 * stage_first_workout / NULLIF(stage_onboarding, 0)), 2)
FROM funnel
UNION ALL
SELECT
    'First Workout → Plan',
    stage_first_workout, stage_plan,
    ROUND(100.0 * stage_plan / NULLIF(stage_first_workout, 0), 2),
    ROUND(100.0 - (100.0 * stage_plan / NULLIF(stage_first_workout, 0)), 2)
FROM funnel;
```

**Salida esperada:**
```
transition                | entered | converted | conversion_rate | drop_off
--------------------------|---------|-----------|----------------|---------
Download → Signup         | 100000  | 65000     | 65.0           | 35.0
Signup → Onboarding       | 65000   | 40000     | 61.54          | 38.46
Onboarding → First Workout| 40000   | 15000     | 37.5           | 62.5
First Workout → Plan      | 15000   | 5000      | 33.33          | 66.67
```

El mayor drop-off está en **Onboarding → First Workout** (62.5%). Ahí hay que intervenir: simplificar el onboarding, agregar un video tutorial, o enviar un push notification apenas termina el setup.

## 5. Cohort Analysis: retención en el tiempo

Un **cohort analysis** agrupa usuarios por una característica común (ej: semana de registro) y mide su comportamiento a lo largo del tiempo. Responde: "los usuarios que se registraron esta semana, ¿vuelven la semana que viene?"

```sql
-- Retention curve por cohort semanal
WITH user_cohort AS (
    SELECT
        user_id,
        DATE_TRUNC('week', signup_date) AS cohort_week
    FROM users
    WHERE signup_date BETWEEN '2026-01-01' AND '2026-06-30'
),
user_activity AS (
    SELECT
        u.user_id,
        u.cohort_week,
        DATE_TRUNC('week', w.workout_date) AS activity_week
    FROM user_cohort u
    JOIN workouts w ON u.user_id = w.user_id
    WHERE w.workout_date >= '2026-01-01'
),
cohort_size AS (
    SELECT cohort_week, COUNT(DISTINCT user_id) AS total_users
    FROM user_cohort
    GROUP BY cohort_week
),
retention AS (
    SELECT
        a.cohort_week,
        a.activity_week,
        COUNT(DISTINCT a.user_id) AS active_users
    FROM user_activity a
    GROUP BY a.cohort_week, a.activity_week
)
SELECT
    r.cohort_week,
    r.activity_week,
    DATEDIFF('week', r.cohort_week, r.activity_week) AS week_number,
    c.total_users,
    r.active_users,
    ROUND(100.0 * r.active_users / c.total_users, 2) AS retention_pct
FROM retention r
JOIN cohort_size c ON r.cohort_week = c.cohort_week
WHERE r.activity_week >= r.cohort_week
AND DATEDIFF('week', r.cohort_week, r.activity_week) <= 8
ORDER BY r.cohort_week, week_number;
```

**Salida esperada:**
```
cohort_week | activity_week | week_number | total_users | active_users | retention_pct
------------|---------------|-------------|-------------|--------------|--------------
2026-05-04  | 2026-05-04    | 0           | 1200        | 1200         | 100.0
2026-05-04  | 2026-05-11    | 1           | 1200        | 480          | 40.0
2026-05-04  | 2026-05-18    | 2           | 1200        | 360          | 30.0
2026-05-04  | 2026-05-25    | 3           | 1200        | 264          | 22.0
2026-05-04  | 2026-06-01    | 4           | 1200        | 204          | 17.0
2026-05-11  | 2026-05-11    | 0           | 1350        | 1350         | 100.0
2026-05-11  | 2026-05-18    | 1           | 1350        | 540          | 40.0
2026-05-11  | 2026-05-25    | 2           | 1350        | 378          | 28.0
```

La cohort `2026-05-11` tiene semana 2 = 28% vs 30% de la cohort anterior. Si lanzaste un cambio de producto el 2026-05-10, este dato sugiere que la retención empeoró. Un cohort analysis como este es la base de cualquier diagnóstico de producto.

## 6. Tipos de cohorts

| Tipo | Agrupación | Pregunta |
|------|------------|----------|
| **Acquisition cohort** | Fecha de registro / instalación | ¿Los usuarios nuevos se comportan distinto a los viejos? |
| **Behavioral cohort** | Acción específica (ej: users que hicieron onboarding vs los que no) | ¿Hacer onboarding mejora retención? |
| **Size-based cohort** | Segmento por volumen (ej: usuarios free vs premium) | ¿Los premium retienen más? |

```sql
-- Behavioral cohort: users que completaron onboarding vs los que no
WITH user_classes AS (
    SELECT
        user_id,
        MAX(CASE WHEN event = 'onboarding_done' THEN 1 ELSE 0 END) AS completed_onboarding
    FROM events
    GROUP BY user_id
),
weekly_retention AS (
    SELECT
        u.user_id,
        u.completed_onboarding,
        DATE_TRUNC('week', signup_date) AS cohort_week,
        DATE_TRUNC('week', w.workout_date) AS activity_week
    FROM user_classes u
    JOIN users us ON u.user_id = us.user_id
    LEFT JOIN workouts w ON u.user_id = w.user_id
)
SELECT
    completed_onboarding,
    DATEDIFF('week', cohort_week, activity_week) AS week_number,
    COUNT(DISTINCT user_id) AS active_users,
    ROUND(100.0 * COUNT(DISTINCT user_id) /
        NULLIF(SUM(COUNT(DISTINCT user_id)) OVER (PARTITION BY completed_onboarding, cohort_week), 0), 2) AS retention_pct
FROM weekly_retention
WHERE activity_week >= cohort_week
AND DATEDIFF('week', cohort_week, activity_week) <= 4
GROUP BY completed_onboarding, cohort_week, activity_week;
```

**Salida esperada:**
```
completed_onboarding | week_number | active_users | retention_pct
--------------------|-------------|-------------|--------------
0 (no onboarding)   | 1           | 2800        | 20.0
0 (no onboarding)   | 2           | 1400        | 10.0
1 (onboarding done) | 1           | 12000       | 60.0
1 (onboarding done) | 2           | 8000        | 40.0
```

El onboarding no es un paso burocrático: los usuarios que lo completan retienen 3× más en semana 2. Esto justifica invertir en mejorar el flujo.

## 7. Visualización de cohorts

Los datos tabulares son difíciles de leer. Un **heatmap de retención** muestra cohorts como filas y semanas como columnas, con color representando el porcentaje.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Datos simulados del cohort query
data = {
    'cohort': ['2026-05-04'] * 5 + ['2026-05-11'] * 5 + ['2026-05-18'] * 5,
    'week':   [0, 1, 2, 3, 4] * 3,
    'retention': [100, 40, 30, 22, 17, 100, 40, 28, 20, 15, 100, 42, 32, 25, 18]
}
df = pd.DataFrame(data)
pivot = df.pivot(index='cohort', columns='week', values='retention')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Retention %'})
plt.title('Cohort Retention Heatmap')
plt.xlabel('Week Number')
plt.ylabel('Cohort Week')
plt.tight_layout()
plt.show()
```

**Salida esperada:**
```
Un heatmap donde las diagonales decrecen (semana 0 = 100%, semana 4 ≈ 15-18%).
Colores más claros en semanas avanzadas indican dónde la retención colapsa.
```

Si ves que cohorts recientes tienen peor retención en semana 1 que cohorts viejas, algo está empeorando. Si una cohort específica muestra una mejora súbita, investigá qué cambió esa semana (feature release, campaña, bug fix).

## Resumen

- **Funnel analysis** mide conversión etapa a etapa e identifica el cuello de botella.
- **Cohort analysis** agrupa usuarios por período o comportamiento y mide retención en el tiempo.
- El mayor drop-off no siempre está al principio: analizá conversión relativa, no absoluta.
- Las behavioral cohorts (ej: onboarding sí vs no) revelan causalidad tentativa.
- Un heatmap de retención condensa semanas de datos en una visualización digerible.
- Los cohorts te permiten separar el efecto de cambios de producto de tendencias estacionales.

## Errores Comunes

- **Definir etapas ambiguas**: "engagement" no es una etapa. Cada etapa debe ser un evento binario y medible.
- **Ignorar el tiempo entre etapas**: un usuario puede tardar 3 días en pasar de signup a onboarding. Si medís todo en el mismo día, perdés usuarios lentos pero valiosos. Usá ventanas de tiempo.
- **Cohorts demasiado pequeñas**: menos de 100 usuarios por cohort y el ruido domina la señal. Agrupá por mes si es necesario.
- **Confundir correlación con causalidad en retención**: la cohort que hizo onboarding retiene más, pero quizás los usuarios motivados completan onboarding y también retienen — el onboarding no es la causa, es un filtro.
- **Funnel sin segmentación**: el funnel promedio es mentira. Segmentá por plataforma (iOS vs Android), país o fuente de adquisición.
- **No considerar survivorship bias**: los usuarios que llegan a la etapa 5 son intrínsecamente diferentes a los que se fueron en etapa 2. No asumas que lo que funciona para ellos funciona para todos.

## Comprueba tu Conocimiento

1. Tu funnel muestra 70% de conversión en Download → Signup pero solo 10% en Onboarding → First Workout. ¿Dónde pondrías los recursos? <!-- En onboarding → first workout. Es el mayor drop-off relativo. Mejorar la conversión de 10% a 20% duplica los usuarios que llegan a plan. -->
2. ¿Qué cohort type usarías para probar si un nuevo tutorial en video mejora retención? <!-- Behavioral cohort: usuarios expuestos al video vs no expuestos. Compará retención semanal entre ambos grupos. -->
3. ¿Por qué un cohort analysis semanal puede ser engañoso si tu producto tiene estacionalidad semanal fuerte (ej: fitness los lunes)? <!-- El día de registro sesga el cohort. Un usuario que se registra un lunes tiene más actividad su primera semana que uno que se registra viernes. Agrupá por día o usá "day of week" como covariable. -->
4. En un heatmap de retención, ¿qué significa que una columna entera se ilumine de repente? <!-- Algo afectó a todas las cohorts por igual en esa semana: un cambio de producto, una campaña de marketing, un bug que infla el conteo, o un evento externo (ej: año nuevo). -->
5. Calculá manualmente: si 5000 users entran a etapa 1, 3000 pasan a etapa 2 y 1500 a etapa 3, ¿cuál es la conversión etapa 1 → etapa 3? <!-- Conversión global = 1500/5000 = 30%. Conversión etapa a etapa: 1→2 = 60%, 2→3 = 50%. -->

## ¿Dónde ir Siguente?

- [[Data Product Thinking]] — cómo elegir la north star metric que tu funnel debe optimizar.
- [[Experimentation & Growth]] — después de diagnosticar, experimentá para mejorar cada etapa.
- [[Advanced Querying]] — más patrones SQL para análisis de producto.
- [[Relational Model & SQL Fundamentals]] — si las window functions te quedan grandes, reforzá bases.
- [[seaborn & Statistical Plots]] — mejores visualizaciones para cohorts y funnels.
- [[Statistics]] — entendé la significancia de diferencias entre cohorts.
- [[Data Warehousing & Lakehouse]] — cómo modelar datos de eventos para que estas queries sean eficientes.
