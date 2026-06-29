---
tags:
  - data-product
  - analytics
  - product-management
  - metrics
  - north-star
status: seedling
created: 2026-06-28
---

# Data Product Thinking

## 1. Escenario de aprendizaje

Tu equipo de producto acaba de lanzar una nueva feature en tu app de fitness: planes de entrenamiento personalizados basados en el historial del usuario, nivel de condición física y objetivos semanales. Tu jefe entra por Slack y te pregunta: "¿cómo medimos si esto fue exitoso?"

No basta con responder "sesiones" o "DAU". Necesitás un framework que conecte la feature con el valor de negocio, que distinga entre métricas vanity y métricas accionables, y que te permita comunicar el impacto a toda la organización.

Este note te va a dar el lenguaje y las herramientas para pensar como un data product manager: qué es un producto de datos, cómo elegir una north star metric y cómo usar frameworks como HEART y AARRR para traducir producto a números.

## 2. Requisitos

- SQL intermedio (agregaciones, JOINs, window functions)
- Familiaridad con producto digital (funnels, retención, cohorts)
- Exposición a dashboards o BI tools

## 3. ¿Qué es un Data Product?

Un **data product** es cualquier activo basado en datos que se consume de manera sistemática para tomar decisiones o impulsar acciones. No es solo un dashboard. Puede ser:

| Tipo | Ejemplo |
|------|---------|
| API de datos | Endpoint que devuelve predicciones de churn |
| Dashboard | Panel de métricas de producto semanal |
| Dataset curado | Tabla `user_health_score` consumida por múltiples equipos |
| Modelo ML | Score de probabilidad de abandono servido en producción |
| Reporte automatizado | Slack bot con KPIs diarios |

Un producto de datos tiene **usuarios**, **SLA**, **dueño**, y **ciclo de vida**. No es un query de una vez.

```sql
-- Un dataset como producto: tabla curada de usuario-health
CREATE TABLE gold.user_health_score AS
SELECT
    u.user_id,
    u.signup_date,
    COALESCE(AVG(w.duration_minutes), 0)  AS avg_workout_duration,
    COALESCE(COUNT(w.workout_id), 0)      AS total_workouts,
    DATEDIFF('day', MAX(w.workout_date), CURRENT_DATE) AS days_since_last_workout,
    CASE
        WHEN COALESCE(COUNT(w.workout_id), 0) = 0                           THEN 'inactive'
        WHEN DATEDIFF('day', MAX(w.workout_date), CURRENT_DATE) > 30        THEN 'churned'
        WHEN AVG(w.duration_minutes) < 15                                   THEN 'low_engagement'
        ELSE 'active'
    END AS health_segment
FROM users u
LEFT JOIN workouts w ON u.user_id = w.user_id
GROUP BY u.user_id, u.signup_date;
```

**Salida esperada:**
```
user_id | signup_date | avg_workout_duration | total_workouts | health_segment
--------|-------------|---------------------|----------------|---------------
1001    | 2026-05-01  | 28.5                | 12             | active
1002    | 2026-05-01  | 0.0                 | 0              | inactive
1003    | 2026-04-15  | 8.2                 | 3              | low_engagement
```

## 4. North Star Metric

La **North Star Metric** es la métrica única que mejor captura el valor duradero que tu producto entrega a los usuarios. Una buena north star:

- **Refleja valor**, no actividad: no es "sesiones", es "minutos de entrenamiento efectivo"
- **Alinea a toda la organización**: producto, marketing, ingeniería, todos reman en la misma dirección
- **Es leading**, no lagging: predecir ingresos, no reportarlos
- **Es accionable**: si sube o baja, sabés qué hacer

| Producto | North Star | Por qué funciona |
|----------|-----------|------------------|
| Spotify | Tiempo escuchando | Captura engagement real |
| Airbnb | Noches reservadas | Correlaciona directo con ingresos y satisfacción |
| Duolingo | Minutos de práctica diaria | Conduce a retención y aprendizaje |
| Fitness app (este caso) | **Workouts completados por semana por usuario** | Refleja el hábito, el valor real |

```sql
-- Calcular north star: weekly workouts per active user
WITH weekly_stats AS (
    SELECT
        user_id,
        DATE_TRUNC('week', workout_date) AS week_start,
        COUNT(DISTINCT workout_id) AS workouts_this_week
    FROM workouts
    WHERE workout_date >= '2026-01-01'
    GROUP BY user_id, DATE_TRUNC('week', workout_date)
)
SELECT
    week_start,
    COUNT(DISTINCT user_id) AS active_users,
    ROUND(AVG(workouts_this_week), 2) AS avg_workouts_per_user,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY workouts_this_week) AS median_workouts
FROM weekly_stats
GROUP BY week_start
ORDER BY week_start;
```

**Salida esperada:**
```
week_start | active_users | avg_workouts_per_user | median_workouts
-----------|-------------|----------------------|----------------
2026-06-01 | 15230       | 3.42                 | 3
2026-06-08 | 15980       | 3.51                 | 4
2026-06-15 | 16450       | 3.28                 | 3
```

## 5. HEART Framework

Google propuso HEART para medir UX a escala. Cada letra es una dimensión:

| Dimensión | Pregunta | Métrica ejemplo |
|-----------|----------|----------------|
| **H**appiness | ¿Están satisfechos? | NPS, CSAT, encuestas in-app |
| **E**ngagement | ¿Usan la feature? | Frecuencia de uso, duración de sesión |
| **A**doption | ¿La descubren? | % de usuarios que probaron la feature |
| **R**etention | ¿Vuelven? | D1/D7/D30 retention de la feature |
| **T**ask Success | ¿Pueden hacer lo que quieren? | Tasa de completación del plan personalizado |

```sql
-- HEART para la feature de planes personalizados
WITH feature_usage AS (
    SELECT
        user_id,
        COUNT(DISTINCT CASE WHEN event = 'plan_generated' THEN event_id END) AS adoption_flag,
        COUNT(DISTINCT CASE WHEN event = 'workout_completed'
                            AND plan_id IS NOT NULL THEN event_id END) AS task_success
    FROM events e
    LEFT JOIN plans p ON e.user_id = p.user_id
    WHERE event_date BETWEEN '2026-06-01' AND '2026-06-30'
    GROUP BY user_id
)
SELECT
    COUNT(*) AS total_users_exposed,
    SUM(CASE WHEN adoption_flag > 0 THEN 1 ELSE 0 END) AS adopted_plan,
    ROUND(100.0 * SUM(CASE WHEN adoption_flag > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS adoption_rate,
    ROUND(AVG(task_success), 2) AS avg_completions_per_user
FROM feature_usage;
```

**Salida esperada:**
```
total_users_exposed | adopted_plan | adoption_rate | avg_completions_per_user
--------------------|-------------|---------------|------------------------
25000               | 18250       | 73.0          | 5.42
```

## 6. Pirate Metrics (AARRR)

El framework AARRR de Dave McClure organiza métricas en el ciclo de vida del usuario:

| Etapa | Pregunta | Ejemplo fitness app |
|-------|----------|-------------------|
| **A**cquisition | ¿Cómo llegan? | Fuente de tráfico, descargas |
| **A**ctivation | ¿Tienen una primera experiencia wow? | Completan setup inicial |
| **R**evenue | ¿Pagan? | Suscripciones premium, ARPU |
| **R**etention | ¿vuelven? | D1/D7/D30 retention del plan |
| **R**eferral | ¿invitan a otros? | Invitaciones enviadas, viral coefficient |

```sql
-- AARRR pipeline semanal
WITH acquisition AS (
    SELECT DATE_TRUNC('week', install_date) AS week, COUNT(*) AS installs
    FROM installs GROUP BY 1
),
activation AS (
    SELECT DATE_TRUNC('week', e.event_date) AS week, COUNT(DISTINCT e.user_id) AS activated
    FROM events e JOIN plans p ON e.user_id = p.user_id
    WHERE e.event = 'first_workout_completed'
    GROUP BY 1
),
revenue AS (
    SELECT DATE_TRUNC('week', payment_date) AS week, SUM(amount) AS mrr
    FROM subscriptions GROUP BY 1
)
SELECT
    a.week,
    a.installs,
    COALESCE(ac.activated, 0) AS activated,
    ROUND(100.0 * COALESCE(ac.activated, 0) / NULLIF(a.installs, 0), 2) AS activation_rate,
    COALESCE(r.mrr, 0) AS mrr
FROM acquisition a
LEFT JOIN activation ac ON a.week = ac.week
LEFT JOIN revenue r ON a.week = r.week
ORDER BY a.week;
```

**Salida esperada:**
```
week       | installs | activated | activation_rate | mrr
-----------|---------|-----------|----------------|------
2026-06-01 | 8500    | 5100      | 60.0           | 45200
2026-06-08 | 9200    | 5888      | 64.0           | 51000
```

## 7. De métricas a acciones: Leading vs Lagging

No todas las métricas son iguales. Clasificarlas te ayuda a saber qué mirar hoy vs qué informar después.

| Característica | Leading | Lagging |
|----------------|---------|---------|
| Definición | Predice el resultado futuro | Informa el resultado pasado |
| Ejemplo | Workouts completados esta semana | Churn rate del mes pasado |
| Acción | Optimizar ahora | Diagnosticar después |
| Plazo | Días / semanas | Semanas / meses |

```sql
-- Leading indicator: engagement semanal como predictor de retention a 30 días
WITH weekly_engagement AS (
    SELECT
        user_id,
        DATE_TRUNC('week', workout_date) AS week,
        COUNT(*) AS workouts
    FROM workouts
    GROUP BY 1, 2
),
retention_flag AS (
    SELECT DISTINCT user_id
    FROM workouts
    WHERE workout_date >= CURRENT_DATE - 30
)
SELECT
    e.week,
    CASE
        WHEN e.workouts >= 4 THEN 'high'
        WHEN e.workouts >= 2 THEN 'medium'
        ELSE 'low'
    END AS engagement_tier,
    COUNT(DISTINCT e.user_id) AS users_in_tier,
    COUNT(DISTINCT r.user_id) AS retained_users,
    ROUND(100.0 * COUNT(DISTINCT r.user_id) / NULLIF(COUNT(DISTINCT e.user_id), 0), 2) AS retention_rate
FROM weekly_engagement e
LEFT JOIN retention_flag r ON e.user_id = r.user_id
WHERE e.week = DATE_TRUNC('week', CURRENT_DATE - 7)
GROUP BY e.week, CASE WHEN e.workouts >= 4 THEN 'high' WHEN e.workouts >= 2 THEN 'medium' ELSE 'low' END;
```

**Salida esperada:**
```
week       | engagement_tier | users_in_tier | retained_users | retention_rate
-----------|----------------|--------------|----------------|---------------
2026-06-21 | high           | 5200         | 4680           | 90.0
2026-06-21 | medium         | 7800         | 5460           | 70.0
2026-06-21 | low            | 4300         | 1290           | 30.0
```

Las **contramétricas** son igual de importantes: si optimizás workouts completados, ¿estás sacrificando seguridad? Una contramétrica para este caso sería *tasa de lesiones reportadas* o *abandono en primera semana por sobrecarga*.

## Resumen

- Un producto de datos tiene usuarios, dueño y ciclo de vida. No es un query de una vez.
- La North Star Metric captura el valor central del producto y alinea equipos.
- HEART (Google) sirve para medir UX en múltiples dimensiones.
- AARRR (Pirate Metrics) cubre el ciclo de vida completo del usuario.
- Leading indicators predicen el futuro; lagging indicators confirman el pasado.
- Las contramétricas evitan que optimices una dimensión a costa de otra.

## Common Mistakes

- **Vanity metrics**: métricas que se ven bien en una slide pero no generan acción (ej: "total de descargas acumuladas").
- **Demasiadas métricas**: más de 5 métricas en un dashboard y nadie sabe cuál mirar.
- **North star que no refleja valor real**: elegir "sesiones" cuando el valor real está en "transacciones completadas".
- **Ignorar contramétricas**: subís engagement pero caés en seguridad o calidad.
- **Métrica única sin segmentación**: el promedio miente. Un promedio de 3 workouts/semana puede esconder 90% de usuarios que hacen 0 y 10% que hacen 30.
- **No conectar métricas a decisiones**: si la métrica sube 5%, ¿qué hacés distinto mañana?

## Check Your Understanding

1. ¿Cuál es la diferencia entre una métrica leading y una lagging? Da un ejemplo de cada una para una app de meditación. <!-- Leading predice (ej: sesiones de meditación completadas por semana predicen retención), lagging confirma (ej: churn rate del trimestre). -->
2. Una app de delivery elige como north star "pedidos por día". ¿Qué problemas ves? <!-- No distingue valor por pedido, no considera calidad del servicio, incentiva pedidos chicos que dañan la economía unitaria. -->
3. ¿Por qué "NPS" funciona mejor como métrica HEART (Happiness) que como north star? <!-- NPS es aspiracional y de encuesta (intermitente), no se puede mover día a día. Una north star debe ser medible continuamente y reflejar uso real. -->
4. Tu north star subió 20%, pero los ingresos cayeron. ¿Qué revisás primero? <!-- Las contramétricas: ¿la feature nueva canibalizó features pagas? ¿usuarios gratuitos consumen más recursos sin convertir? -->
5. Aplicá AARRR a una app de citas. ¿Qué métrica pondrías en cada etapa? <!-- Acquisition: descargas; Activation: primer match/mensaje; Revenue: suscripciones; Retention: D7; Referral: invites enviados. -->

## Where to Go Next

- [[Funnel & Cohort Analysis]] — profundizá en cómo diagnosticar pérdida de usuarios etapa por etapa.
- [[Experimentation & Growth]] — conectá métricas con experimentos y growth loops.
- [[Relational Model & SQL Fundamentals]] — si necesitás reforzar SQL para implementar estas métricas.
- [[Visualization Fundamentals]] — cómo graficar estas métricas sin mentir.
- [[Stakeholder Communication]] — cómo presentar métricas a la C-suite sin que te maten.
- [[Self-Serve Analytics]] — cómo escalar estas métricas para que todo el equipo las consuma.
- [[A-B Testing]] — cómo probar que tu feature realmente movió la north star.
