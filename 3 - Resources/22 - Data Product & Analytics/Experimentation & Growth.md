---
tags:
  - experimentation
  - growth
  - a-b-testing
  - product-analytics
  - feature-flags
status: seedling
created: 2026-06-28
---

# Experimentation & Growth

## 1. Escenario de aprendizaje

Tu equipo quiere rediseñar el onboarding de la app de fitness: menos pasos, más video, una pregunta de objetivo al inicio. Suena bien, pero si el nuevo onboarding empeora la retención, arruinaste la experiencia de todos los usuarios nuevos durante semanas.

No podés lanzar al 100% sin saber. Necesitás **experimentación estructurada**: feature flags, asignación aleatoria, tamaño muestral calculado, y un análisis que distinga señal de ruido.

Además, el crecimiento no es solo experimentar: es diseñar **growth loops** que hagan que el producto crezca orgánicamente. Este note conecta experimentación con growth engineering.

## 2. Requisitos

- SQL intermedio (CTEs, JOINs, agregaciones condicionales)
- Estadística básica: media, varianza, desviación estándar, test de hipótesis
- Concepto de feature flags y gradual rollout

## 3. Feature Flags

Un **feature flag** (o toggle) es un interruptor que activa/desactiva una feature sin hacer deploy. Permite:

- **Canary release**: lanzar a 1%, 5%, 10% progresivamente
- **Kill switch**: apagar la feature al instante si algo sale mal
- **Targeted rollout**: activar para usuarios internos, beta testers, o por país
- **A/B testing**: asignar users a control vs tratamiento

```sql
-- Asignación de users a grupos de feature flag
WITH user_assignment AS (
    SELECT
        user_id,
        CASE
            WHEN MOD(user_id, 100) < 50 THEN 'control'
            ELSE 'treatment'
        END AS experiment_group
    FROM users
    WHERE signup_date >= '2026-06-01'
),
experiment_events AS (
    SELECT
        u.user_id,
        u.experiment_group,
        COUNT(DISTINCT w.workout_id) AS workouts,
        SUM(CASE WHEN w.workout_date <= u.signup_date + 7 THEN 1 ELSE 0 END) AS week1_workouts
    FROM user_assignment u
    LEFT JOIN workouts w ON u.user_id = w.user_id
    GROUP BY u.user_id, u.experiment_group
)
SELECT
    experiment_group,
    COUNT(*) AS users,
    ROUND(AVG(workouts), 2) AS avg_workouts,
    ROUND(AVG(week1_workouts), 2) AS avg_week1_workouts,
    ROUND(STDDEV(week1_workouts), 2) AS std_week1_workouts
FROM experiment_events
GROUP BY experiment_group;
```

**Salida esperada:**
```
experiment_group | users | avg_workouts | avg_week1_workouts | std_week1_workouts
----------------|-------|-------------|-------------------|-------------------
control         | 12500 | 4.2         | 1.8               | 2.1
treatment       | 12480 | 4.8         | 2.4               | 2.3
```

El tratamiento muestra +0.6 workouts promedio en semana 1. Pero antes de celebrar, necesitamos un test estadístico.

## 4. Experiment Design: hipótesis, métrica, muestra, duración

Diseñar un experimento requiere 4 decisiones previas:

| Componente | Definición | Ejemplo |
|------------|------------|---------|
| **Hipótesis** | Lo que creés que va a pasar | "El nuevo onboarding aumenta workouts en semana 1 en al menos 15%" |
| **Métrica primaria** | La métrica que decide el ganador | Workouts completados en días 1-7 |
| **Tamaño muestral** | Cuántos usuarios necesitás por grupo | 12.000 por grupo (para MDE = 10%, α = 0.05, β = 0.20) |
| **Duración** | Cuántos días correr el experimento | 14 días (cubrir al menos un ciclo semanal completo) |

```sql
-- Cálculo de tamaño muestral aproximado en SQL (usando fórmula de diferencia de proporciones)
-- Asumimos tasa de conversión base p1 = 0.20 y MDE absoluto de 0.02
WITH params AS (
    SELECT
        0.20 AS p1,
        0.02 AS mde,
        1.96 AS z_alpha,   -- α = 0.05 (two-tailed)
        0.84 AS z_beta     -- β = 0.20 (80% power)
)
SELECT
    CEIL(
        (z_alpha + z_beta) * (z_alpha + z_beta) *
        (p1 * (1 - p1) + (p1 + mde) * (1 - (p1 + mde))) /
        (mde * mde)
    ) AS sample_size_per_group
FROM params;
```

**Salida esperada:**
```
sample_size_per_group
---------------------
9800
```

Necesitás ~9.800 usuarios por grupo para detectar un cambio absoluto de 2 puntos porcentuales en conversión. Si tu app genera 2.000 signups/día, necesitás ~10 días de experimento.

```sql
-- Análisis post-experimento: test Z de proporciones
WITH experiment_results AS (
    SELECT
        experiment_group,
        COUNT(*) AS total_users,
        SUM(CASE WHEN week1_workouts >= 2 THEN 1 ELSE 0 END) AS retained_users
    FROM experiment_events
    GROUP BY experiment_group
),
control_stats AS (
    SELECT total_users AS n_c, retained_users AS x_c
    FROM experiment_results WHERE experiment_group = 'control'
),
treatment_stats AS (
    SELECT total_users AS n_t, retained_users AS x_t
    FROM experiment_results WHERE experiment_group = 'treatment'
)
SELECT
    x_c AS control_conversions,
    n_c AS control_total,
    ROUND(100.0 * x_c / n_c, 2) AS control_rate,
    x_t AS treatment_conversions,
    n_t AS treatment_total,
    ROUND(100.0 * x_t / n_t, 2) AS treatment_rate,
    ROUND(100.0 * (1.0 * x_t / n_t - 1.0 * x_c / n_c) / (1.0 * x_c / n_c), 2) AS relative_lift,
    CASE
        WHEN ABS(1.0 * x_t / n_t - 1.0 * x_c / n_c) >
             1.96 * SQRT(
                 (1.0 * (x_c + x_t) / (n_c + n_t)) *
                 (1 - 1.0 * (x_c + x_t) / (n_c + n_t)) *
                 (1.0 / n_c + 1.0 / n_t)
             ) THEN 'Statistically Significant'
        ELSE 'Not Significant'
    END AS significance
FROM control_stats, treatment_stats;
```

**Salida esperada:**
```
control_conversions | control_total | control_rate | treatment_conversions | treatment_total | treatment_rate | relative_lift | significance
-------------------|--------------|-------------|---------------------|----------------|---------------|--------------|-------------
2500               | 12500        | 20.0        | 2995                 | 12480          | 24.0          | 20.0         | Statistically Significant
```

Un 20% de lift relativo con significancia estadística. Ahora sí podés presentar resultados al equipo.

## 5. Growth Loops

Un **growth loop** es un sistema autosostenido donde la salida de una iteración es la entrada de la siguiente. A diferencia de los funnels (lineales), los loops son circulares.

| Loop | Cómo funciona | Ejemplo clásico |
|------|--------------|-----------------|
| **Viral loop** | Usuario invita → invitado se registra → nuevo usuario invita | Dropbox (espacio extra por referral) |
| **Content loop** | Usuario crea contenido → contenido atrae más usuarios → nuevos usuarios crean más contenido | YouTube, Airbnb (listings atraen huéspedes) |
| **Paid loop** | Ingreso por usuario → se reinvierte en ads → ad atrae más usuarios | Uber, Peloton |

```sql
-- Medir el viral coefficient: cuántos nuevos usuarios trae cada usuario existente
WITH referral_data AS (
    SELECT
        r.referrer_user_id,
        COUNT(DISTINCT r.referred_user_id) AS referrals_sent,
        COUNT(DISTINCT CASE WHEN u.signup_date IS NOT NULL THEN r.referred_user_id END) AS referrals_converted
    FROM referrals r
    LEFT JOIN users u ON r.referred_user_id = u.user_id
    WHERE r.referral_date BETWEEN '2026-06-01' AND '2026-06-30'
    GROUP BY r.referrer_user_id
)
SELECT
    ROUND(AVG(referrals_sent), 2) AS avg_referrals_per_user,
    ROUND(AVG(referrals_converted), 2) AS avg_converted_referrals_per_user,
    ROUND(1.0 * SUM(referrals_converted) / NULLIF(SUM(referrals_sent), 0), 2) AS conversion_rate,
    CASE
        WHEN AVG(referrals_converted) >= 1.0 THEN 'Viral (v >= 1)'
        ELSE 'Not viral (v < 1)'
    END AS viral_status
FROM referral_data;
```

**Salida esperada:**
```
avg_referrals_per_user | avg_converted_referrals_per_user | conversion_rate | viral_status
----------------------|-------------------------------|----------------|-------------
0.8                   | 0.24                          | 0.30           | Not viral (v < 1)
```

El viral coefficient v = 0.24. Cada usuario trae 0.24 nuevos usuarios. Necesitamos v >= 1 para crecimiento orgánico sostenido. Esto significa que el loop necesita un boost (mejorar incentivo, reducir fricción en el invite).

## 6. North Star → Growth Model

La **north star metric** se descompone en **palancas de crecimiento**. Cada palanca es un input que podes optimizar mediante experimentos.

```
North Star: Workouts completados por semana
                │
        ┌───────┼───────────┐
        │       │           │
   Usuarios  Frecuencia  Duración
   activos   por user    por sesión
        │       │           │
   Retención  Engagement  Profundidad
   (cohort)   (DAU/MAU)   (minutos)
```

Cada palanca se traduce a experimentos:

```sql
-- Descomposición de north star en palancas
WITH north_star_decomp AS (
    SELECT
        DATE_TRUNC('week', w.workout_date) AS week,
        COUNT(DISTINCT w.user_id) AS active_users,
        COUNT(DISTINCT w.workout_id) * 1.0 / NULLIF(COUNT(DISTINCT w.user_id), 0) AS freq_per_user,
        AVG(w.duration_minutes) AS avg_duration
    FROM workouts w
    WHERE w.workout_date >= '2026-01-01'
    GROUP BY 1
)
SELECT
    week,
    active_users,
    ROUND(freq_per_user, 2) AS frequency,
    ROUND(avg_duration, 2) AS avg_duration,
    ROUND(active_users * freq_per_user * avg_duration, 0) AS estimated_total_workout_minutes
FROM north_star_decomp
ORDER BY week;
```

**Salida esperada:**
```
week       | active_users | frequency | avg_duration | estimated_total_minutes
-----------|-------------|-----------|-------------|-----------------------
2026-06-01 | 15230       | 3.42      | 28.5        | 1485074
2026-06-08 | 15980       | 3.51      | 29.2        | 1637438
```

Si querés crecer la north star 10%, podés atacar cualquiera de las 3 palancas. Un experimento en onboarding apunta típicamente a active_users (retención) y frequency.

## 7. Análisis de experimentos: CUPED, peeking, sequential testing

El análisis ingenuo de experimentos tiene trampas. Tres conceptos clave:

### CUPED (Controlled-experiment Using Pre-Experiment Data)
Usa datos pre-experimento para reducir la varianza y detectar efectos más chicos con la misma muestra.

```sql
-- CUPED: ajustar métrica post usando correlación con período pre-experimento
WITH pre_experiment AS (
    SELECT
        user_id,
        COUNT(*) AS pre_workouts
    FROM workouts
    WHERE workout_date BETWEEN '2026-05-01' AND '2026-05-31'
    GROUP BY user_id
),
experiment AS (
    SELECT
        u.user_id,
        u.experiment_group,
        COUNT(DISTINCT w.workout_id) AS post_workouts
    FROM user_assignment u
    JOIN workouts w ON u.user_id = w.user_id
    WHERE w.workout_date BETWEEN '2026-06-01' AND '2026-06-14'
    GROUP BY u.user_id, u.experiment_group
)
SELECT
    e.experiment_group,
    ROUND(AVG(e.post_workouts), 2) AS raw_mean,
    ROUND(AVG(e.post_workouts) - 0.3 * (AVG(p.pre_workouts) - 5.0), 2) AS cuped_adjusted_mean
FROM experiment e
LEFT JOIN pre_experiment p ON e.user_id = p.user_id
GROUP BY e.experiment_group;
```

**Salida esperada:**
```
experiment_group | raw_mean | cuped_adjusted_mean
----------------|---------|-------------------
control         | 4.2      | 4.15
treatment       | 4.8      | 4.82
```

CUPED redujo la diferencia de medias de 0.6 a 0.67 — el efecto es más claro después de ajustar por ruido pre-experimento.

### Peeking y Sequential Testing
Mirar el experimento todos los días y detenerlo cuando el p-value cruza 0.05 **infla** la tasa de falsos positivos. Usá _sequential testing_ (ej: siempre válido, peeking libre con alpha spending).

## Resumen

- **Feature flags** permiten rollout gradual y kill switches. Son la infraestructura base de experimentación.
- Un **experimento bien diseñado** tiene hipótesis, métrica primaria, tamaño muestral y duración pre-registrados.
- Los **growth loops** (viral, content, paid) son motores de crecimiento autosostenido. Medí el viral coefficient.
- La **north star** se descompone en palancas. Cada experimento ataca una palanca.
- **CUPED** reduce varianza usando datos pre-experimento.
- **Peeking** sin corrección invalida tus conclusiones. Usá sequential testing.

## Common Mistakes

- **Experimento sin poder estadístico**: terminás con conclusión "no significativo" cuando el efecto existe pero no tenías suficientes usuarios.
- **Múltiples tests sin corrección**: si mirás 20 métricas secundarias, una va a ser "significativa" por azar. Aplicá Bonferroni o Benjamini-Hochberg.
- **Ignorar el efecto novelty**: los usuarios interactúan más con lo nuevo simplemente porque es nuevo. La métrica puede volver a la media después de 2-3 semanas.
- **Detener el experimento temprano**: la varianza necesita tiempo para estabilizarse. Un mínimo de 1-2 ciclos semanales completos (14 días).
- **Contaminación entre grupos**: si control y tratamiento son usuarios de la misma app y se enteran de la diferencia, el comportamiento cambia.
- **Selección de métrica post-hoc**: primero definís qué medís, después analizás. Elegir la métrica después de ver los resultados es p-hacking.

## Check Your Understanding

1. Diseñá un experimento para probar si un push de "¡Vamos! Hora de entrenar" aumenta workouts semanales. Especificá hipótesis, métrica primaria, y duración. <!-- Hipótesis: push aumenta workouts semanales en 10%. Métrica: workouts completados en los 7 días post-push. Duración: 14 días (2 semanas, cubriendo efecto novelty). -->
2. ¿Por qué no basta con un p-value < 0.05 para declarar ganador? <!-- Podría ser falso positivo por múltiples tests, efecto novelty, o sesgo de segmentación. También necesitás significancia práctica (effect size mínimo relevante). -->
3. Tu viral coefficient es 0.8. ¿Crecés orgánicamente? <!-- No. v < 1 significa que cada usuario no reemplaza ni siquiera un usuario nuevo. Necesitás v >= 1 para crecimiento viral. -->
4. ¿Qué es peeking y por qué es peligroso? <!-- Mirar el p-value repetidamente y detener cuando cruza 0.05. Infla la tasa de falsos positivos porque cada mirada es una oportunidad extra de cruzar el umbral por azar. -->
5. Si el tratamiento mejora la métrica primaria 15% pero empeora el NPS 5 puntos, ¿qué hacés? <!-- No hay respuesta automática. Depende de la prioridad del negocio. Lo correcto es reportar ambos efectos y dejar la decisión al PM. Por eso siempre medís contramétricas. -->

## Where to Go Next

- [[Experimental Design]] — diseño de experimentos más riguroso: aleatorización, blocking, factorial designs.
- [[A-B Testing]] — implementación técnica de A/B tests en la práctica.
- [[Data Product Thinking]] — conectá experimentos con la north star metric.
- [[Funnel & Cohort Analysis]] — diagnosticá qué etapa del funnel optimizar con experimentos.
- [[Statistics]] — profundizá en tests estadísticos, power analysis, y Bayesian methods.
- [[CLI & Productivity]] — herramientas para automatizar pipelines de experimentación.
- [[Feature Stores]] — cómo servir features de usuarios para segmentación y CUPED a escala.
