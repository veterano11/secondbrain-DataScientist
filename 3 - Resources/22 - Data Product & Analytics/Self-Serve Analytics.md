---
tags:
  - self-serve
  - analytics
  - bi
  - data-governance
  - data-warehouse
  - semantic-layer
status: seedling
created: 2026-06-28
---

# Self-Serve Analytics

## 1. Escenario de aprendizaje

El equipo de producto te escribe 10 veces por semana en Slack preguntando métricas simples:

- "¿Cuántos usuarios se registraron ayer?"
- "¿Cuál es el conversion rate de free a premium este mes?"
- "¿Cómo va la retention de la cohort de junio?"

No es malicia ni flojera. No tienen acceso directo a los datos, no saben SQL, y las herramientas que tienen (un dashboard PDF semanal) están desactualizados el día después de generarse.

**Self-serve analytics** es la práctica de empoderar a equipos no técnicos para responder sus propias preguntas de datos, con las herramientas y el conocimiento adecuados, sin depender del equipo de datos para cada consulta.

Este note cubre la infraestructura (semantic layer, BI tools, metrics store), la habilitación (data literacy, training, playbooks) y la gobernanza (accesos, certificación, single source of truth).

## 2. Requisitos

- SQL intermedio (JOINs, agregaciones, CTEs)
- Familiaridad con dbt o herramientas de transformación
- Conocimiento básico de BI tools (Metabase, Superset, Looker)
- Exposición a conceptos de data warehouse (schemas, modelos de datos)

## 3. Semantic Layer

El **semantic layer** es una capa de abstracción entre los datos crudos y el usuario final. Define métricas y dimensiones de manera consistente para que todos los equipos calculen "ingresos" de la misma forma.

Sin semantic layer, cada equipo define sus métricas ad-hoc:

```sql
-- Equipo de producto: ingresos = suscripciones nuevas
SELECT COUNT(*) FROM subscriptions WHERE status = 'active';

-- Equipo de finanzas: ingresos = cobros confirmados
SELECT SUM(amount) FROM payments WHERE status = 'settled';

-- Equipo de growth: ingresos = bookings
SELECT SUM(amount) FROM invoices WHERE created_at >= '2026-01-01';
```

Tres números distintos. Todos creen tener la razón. El semantic layer resuelve esto.

```sql
-- Semantic layer en dbt: definición canónica de revenue
WITH revenue AS (
    SELECT
        DATE_TRUNC('day', p.settled_at) AS date_day,
        p.user_id,
        p.amount,
        'subscription' AS revenue_type
    FROM payments p
    WHERE p.status = 'settled'

    UNION ALL

    SELECT
        DATE_TRUNC('day', o.completed_at) AS date_day,
        o.user_id,
        o.amount,
        'one_time' AS revenue_type
    FROM orders o
    WHERE o.status = 'completed'
)
SELECT
    date_day,
    user_id,
    SUM(amount) AS total_revenue,
    revenue_type,
    COUNT(*) AS transaction_count
FROM revenue
GROUP BY date_day, user_id, revenue_type;
```

**Salida esperada:**
```
date_day   | user_id | total_revenue | revenue_type   | transaction_count
-----------|--------|--------------|----------------|-----------------
2026-06-28 | 1001   | 29.99        | subscription   | 1
2026-06-28 | 1002   | 59.99        | one_time       | 1
```

Cualquier equipo que use esta tabla obtiene el mismo número para "ingresos". El semantic layer está implementado en el modelo de datos (dbt, LookML, o directamente en la vista SQL).

## 4. Metrics Store

Un **metrics store** es un repositorio central donde se definen, documentan y versionan las métricas de la organización. Herramientas como dbt Metrics, Metriql, o Transform permiten:

- Definir métricas en YAML o SQL
- Versionarlas con git
- Exponerlas via API a BI tools
- Mantener descripciones, dueños, y definiciones

```yaml
# metrics_store.yml — definición centralizada de métricas
metrics:
  - name: monthly_active_users
    label: Monthly Active Users
    description: Usuarios únicos con al menos un workout en los últimos 30 días
    model: ref('user_activity')
    calculation_method: count_distinct
    expression: user_id
    filters:
      - field: has_workout
        operator: '='
        value: true
    time_grains: [day, week, month]
    owner: data_team

  - name: revenue_monthly
    label: Monthly Recurring Revenue
    description: Suma de pagos confirmados por suscripción en el mes
    model: ref('revenue')
    calculation_method: sum
    expression: total_revenue
    filters:
      - field: revenue_type
        operator: '='
        value: subscription
    time_grains: [day, month]
    owner: finance_team
```

```sql
-- Usando la métrica definida en el metrics store
-- dbt metrics query generado automáticamente
SELECT
    DATE_TRUNC('month', date_day) AS month,
    COUNT(DISTINCT user_id) AS monthly_active_users
FROM {{ ref('user_activity') }}
WHERE has_workout = TRUE
  AND date_day BETWEEN '2026-01-01' AND '2026-06-30'
GROUP BY 1
ORDER BY 1;
```

**Salida esperada:**
```
month      | monthly_active_users
-----------|--------------------
2026-01-01 | 18500
2026-02-01 | 19200
2026-03-01 | 20500
2026-04-01 | 19800
2026-05-01 | 21300
2026-06-01 | 22500
```

Con un metrics store, el equipo de producto puede consultar MAU sin escribir SQL desde cero — solo seleccionan la métrica y el período en la BI tool.

## 5. BI Tools: comparación

La elección de la BI tool depende del perfil del equipo y la infraestructura existente.

| Herramienta | Tipo | Curva de aprendizaje | Semantic layer nativo | Ideal para |
|------------|------|--------------------|---------------------|-----------|
| **Metabase** | SQL + drag-drop | Baja | No (usa SQL directo) | Equipos chicos, startups |
| **Apache Superset** | SQL + chart builder | Media | Parcial | Equipos con infra propia |
| **Looker / Looker Studio** | LookML + SQL | Alta | Sí (LookML) | Organizaciones grandes |
| **Tableau** | Drag-drop | Media | No (conexión directa) | Analistas, power users |

```sql
-- Creando una vista en Superset que el equipo de producto puede consultar sin SQL
-- Esta vista expone datos listos para drag-drop
CREATE VIEW self_serve.daily_kpis AS
SELECT
    dt.date_day,
    COUNT(DISTINCT u.user_id) AS new_users,
    COUNT(DISTINCT w.user_id) AS active_users,
    ROUND(COUNT(DISTINCT s.user_id) * 100.0 / NULLIF(COUNT(DISTINCT u.user_id), 0), 2) AS conversion_rate
FROM (
    SELECT DISTINCT DATE_TRUNC('day', series) AS date_day
    FROM generate_series('2026-01-01'::date, '2026-06-30'::date, '1 day'::interval) AS series
) dt
LEFT JOIN users u ON DATE_TRUNC('day', u.signup_date) = dt.date_day
LEFT JOIN workouts w ON DATE_TRUNC('day', w.workout_date) = dt.date_day
LEFT JOIN subscriptions s ON DATE_TRUNC('day', s.subscription_date) = dt.date_day
    AND s.user_id IN (SELECT user_id FROM users WHERE signup_date = dt.date_day)
GROUP BY dt.date_day;
```

**Salida esperada:**
```
date_day   | new_users | active_users | conversion_rate
-----------|----------|-------------|----------------
2026-06-01 | 350      | 5200        | 5.2
2026-06-02 | 280      | 5100        | 5.5
```

Esta vista se conecta a Metabase o Superset y permite al equipo de producto construir sus propios filtros y gráficos sin escribir SQL.

## 6. Data Literacy

La mejor herramienta del mundo no sirve si el equipo no sabe qué preguntar ni cómo interpretar los resultados. **Data literacy** es la habilidad de leer, entender, crear y comunicar datos como información.

Para implementar self-serve necesitás:

### Trainings
- **Nivel 1** — Leer dashboards: qué significa cada KPI, cómo usar filtros, cómo exportar
- **Nivel 2** — Explorar datos: usar drag-drop en Metabase, construir tablas simples, entender promedios vs medianas
- **Nivel 3** — SQL básico: SELECT, WHERE, GROUP BY para responder preguntas ad-hoc

### Playbooks de análisis
Documentación de los análisis más comunes que el equipo puede consultar:

```markdown
## Playbook: ¿Cómo medir el impacto de una campaña de push notifications?

1. Ir a Metabase > Dashboard > Push Campaigns
2. Seleccionar fecha de campaña en el filtro
3. Comparar conversión y retention de usuarios que recibieron push vs control
4. Si conversion_rate > 5% y retention_diff > 10%, la campaña fue exitosa
5. Exportar a PDF y compartir en Slack #analytics
```

### Documentación de métricas
Cada métrica en el metrics store debe tener:

- **Nombre**: Monthly Active Users
- **Definición**: Usuarios únicos con >= 1 workout en los últimos 30 días
- **Fórmula**: COUNT(DISTINCT user_id) WHERE datediff('day', last_workout, today) <= 30
- **Dueño**: data_team
- **Frescuridad**: diaria (se actualiza cada 6 AM)
- **Interpretación**: Si MAU baja 5% semana a semana, revisar retention de nuevas cohorts

## 7. Governance en self-serve

Self-serve no significa "que cada uno haga lo que quiera". Sin gobernanza, terminás con 5 definiciones distintas de "ingresos" y nadie confía en los datos.

### Principios de governance

| Principio | Implementación |
|-----------|---------------|
| **Single source of truth** | Una tabla de hechos por dominio, certificada por el data team |
| **Accesos basados en roles** | Lectura a todos, escritura solo a data team |
| **Certificación** | Datasets certificados tienen un sello visual en la BI tool |
| **Linaje** | Cada métrica muestra su origen (dbt docs, columna, transformación) |
| **Alertas** | Si un pipeline falla, se notifica a todos los consumidores del dataset |

```sql
-- Query de governance: ¿quién usa qué datasets?
SELECT
    query_username,
    table_schema || '.' || table_name AS dataset,
    COUNT(*) AS query_count,
    MIN(query_start_time) AS first_seen,
    MAX(query_start_time) AS last_seen
FROM query_history
WHERE table_schema IN ('self_serve', 'gold', 'public')
  AND query_start_time >= CURRENT_DATE - 30
GROUP BY query_username, dataset
ORDER BY query_count DESC;
```

**Salida esperada:**
```
query_username | dataset                   | query_count | first_seen         | last_seen
---------------|--------------------------|-------------|-------------------|-------------------
pm_maria      | self_serve.daily_kpis     | 142         | 2026-06-01 09:00  | 2026-06-28 14:30
pm_juan       | self_serve.user_segments  | 85          | 2026-06-03 11:15  | 2026-06-28 10:00
pm_maria      | self_serve.funnel_weekly  | 63          | 2026-06-05 08:30  | 2026-06-28 12:00
```

Este query te dice quién está usando qué. Si `daily_kpis` tiene 3 consumidores activos, sabés que su certificación es crítica.

## 8. Common Mistakes

- **No gobernar métricas**: cada equipo define "ingresos" distinto. El resultado: desconfianza generalizada en los datos.
- **Dar acceso sin training**: le das Metabase al equipo de producto sin explicar cómo usarlo. Terminan descargando CSVs y haciéndolo en Excel.
- **BI sin semantic layer**: conectás Tableau directo a la base de datos. Cada dashboard tiene JOINs distintos. Performance se degrada. Nadie sabe qué métrica es correcta.
- **Dashboards sin contexto**: un número solo no significa nada. Sin benchmarks (vs semana pasada, vs meta, vs cohort similar), el equipo no sabe si 5% es bueno o malo.
- **Frescura no documentada**: el equipo ve datos de ayer y no sabe que el pipeline corre cada 24 horas y todavía no se actualizó. Toman decisiones con datos incompletos.
- **Sobre-ingeniería**: crear un semantic layer complejísimo cuando el equipo solo necesita 10 métricas. Empezá por las métricas más preguntadas y expandí desde ahí.

## Resumen

- **Self-serve analytics** empodera al equipo de producto para responder sus propias preguntas, liberando al data team para análisis más profundos.
- El **semantic layer** asegura que todos calculen las métricas de la misma forma y desde la misma fuente.
- Un **metrics store** centraliza definiciones, dueños y versiones de métricas.
- La **BI tool** debe elegirse según el perfil del equipo, no según la preferencia del data team.
- **Data literacy** (trainings + playbooks) es tan importante como la herramienta. Sin ella, no hay self-serve real.
- La **governance** es condición necesaria: accesos, certificación y linaje evitan el caos.
- Empezá chico: resolvé las 10 preguntas más frecuentes antes de construir el semantic layer perfecto.

## Errores Comunes

- **Dar acceso a datos crudos**: el equipo de producto no debería tocar tablas de staging ni raw. Exponé solo tablas curadas (gold layer).
- **No tener un feedback loop**: self-serve es bidireccional. El equipo necesita un canal para reportar datos incorrectos o métricas faltantes.
- **Dashboards lentos**: si un dashboard tarda más de 5 segundos en cargar, el equipo no lo usa. Optimizá agregaciones pre-calculadas.
- **Ignorar la estacionalidad**: el equipo compara el lunes con el domingo y cree que algo explotó. Educá sobre comparaciones like-for-like (WoW, YoY).

## Comprueba tu Conocimiento

1. ¿Cuál es la diferencia entre semantic layer y metrics store? <!-- Semantic layer es la implementación técnica (vistas SQL, LookML, tablas gold) que unifica la definición de métricas. Metrics store es el repositorio de metadata que documenta y versiona esas definiciones. -->
2. Tu equipo de producto usa Metabase. ¿Cómo implementás un semantic layer sin LookML? <!-- Creando vistas gold en la base de datos (schemas self_serve o gold) que exponen métricas pre-calculadas con JOINs resueltos. Metabase se conecta a esas vistas como si fueran tablas. -->
3. ¿Por qué dar acceso SQL directo puede ser contraproducente para self-serve? <!-- El equipo escribe queries ineficientes que degradan la base de producción, obtienen resultados incorrectos por JOINs mal hechos, y generan desconfianza. -->
4. El equipo de producto te pide 5 métricas nuevas por semana. ¿Cuál es tu estrategia? <!-- Implementar un proceso: (1) documentar la métrica en el metrics store, (2) priorizar según frecuencia de consulta, (3) si es una métrica que ya existe con otro nombre, educar al equipo en lugar de duplicar. -->
5. ¿Qué métrica monitoreás para saber si tu self-serve strategy funciona? <!-- Número de consultas del equipo de producto sin intervención del data team, tiempo desde que preguntan hasta que obtienen respuesta, y satisfacción con los datos (encuesta trimestral). -->

## ¿Dónde ir Siguente?

- [[dbt & Data Transformation]] — cómo construir el semantic layer con dbt.
- [[Data Warehousing & Lakehouse]] — arquitectura de datos que soporta self-serve (medallion architecture: bronze → silver → gold).
- [[Data Product Thinking]] — conectá self-serve con la filosofía de datos como producto.
- [[Stakeholder Communication]] — cómo presentar el ROI de self-serve a la C-suite.
- [[Data Quality & Testing]] — cómo asegurar que las métricas self-serve sean confiables.
- [[Relational Model & SQL Fundamentals]] — si necesitás reforzar SQL para construir las vistas gold.
- [[Data Classification & Governance]] — políticas de acceso, PII, y cumplimiento en self-serve.
