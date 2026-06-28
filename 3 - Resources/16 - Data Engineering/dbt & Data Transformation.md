---
tags:
  - data-engineering
  - dbt
  - data-transformation
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tus transforms de SQL están en scripts de Python que concatenan strings. Cada nueva columna requiere cambiar código frágil, no hay tests, la documentación está en la cabeza de una persona, y replicar el pipeline en otro entorno es imposible. [[dbt & Data Transformation|dbt]] (data build tool) permite escribir transforms como consultas SQL con tests, documentación y versionado, justo lo que necesitas para que tu pipeline sea mantenible.

## 1. ¿Qué es dbt?

dbt es una herramienta que convierte **SQL en código**: escribes modelos (archivos `.sql` con `SELECT`), y dbt se encarga del DDL (`CREATE TABLE`, `CREATE VIEW`), la compilación, la documentación y la ejecución en orden de dependencia.

```sql
-- models/ventas_por_cliente.sql
-- dbt compila este SELECT y crea una tabla o vista
SELECT
    c.cliente_id,
    c.nombre,
    SUM(v.monto) AS total_gastado,
    COUNT(DISTINCT v.orden_id) AS ordenes
FROM {{ source('raw', 'clientes') }} c
JOIN {{ ref('ventas_limpias') }} v ON c.cliente_id = v.cliente_id
GROUP BY 1, 2

-- Salida esperada: dbt compila a:
-- CREATE TABLE analytics.ventas_por_cliente AS
-- SELECT c.cliente_id, ...
```

**Materializaciones**: cómo dbt persiste el modelo:
- `view`: solo guarda el `CREATE VIEW` (rápido, siempre datos frescos).
- `table`: `CREATE TABLE ... AS SELECT` (datos snapshot, más rápido en consultas).
- `incremental`: solo procesa datos nuevos (ideal para tablas grandes).

## 2. Modelos

Un modelo es un archivo `.sql` que hace un `SELECT`. Usa dos funciones clave:

- `{{ ref('otro_modelo') }}`: referencia a otro modelo dbt → dbt resuelve el orden de ejecución automáticamente.
- `{{ source('sistema', 'tabla') }}`: referencia a una tabla raw definida en `sources.yml`.

```sql
-- models/ventas_limpias.sql
{{ config(materialized='incremental', unique_key='venta_id') }}

SELECT
    venta_id,
    cliente_id,
    producto_id,
    monto,
    fecha
FROM {{ source('oltp', 'ventas') }}
{% if is_incremental() %}
    WHERE fecha > (SELECT MAX(fecha) FROM {{ this }})
{% endif %}

-- Salida esperada:
-- Primera ejecución: CREATE TABLE ventas_limpias AS SELECT * FROM oltp.ventas
-- Ejecuciones siguientes: INSERT INTO ventas_limpias SELECT * FROM oltp.ventas WHERE fecha > max_fecha
```

## 3. Tests

dbt incorpora testing de datos como parte del flujo normal:

- **Generic tests**: declarativos en `schema.yml`.
  - `not_null`, `unique`, `accepted_values`, `relationships`
- **Singular tests**: consultas SQL que deben devolver 0 filas. Si devuelven alguna, el test falla.

```yaml
# schema.yml
version: 2
models:
  - name: ventas_limpias
    columns:
      - name: venta_id
        tests:
          - unique
          - not_null
      - name: monto
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
    tests:
      - relationships:
          to: ref('clientes')
          field: cliente_id
```

## 4. Documentación

dbt genera un sitio de documentación automáticamente:

```bash
dbt docs generate    # compila docs de modelos, columnas, tests, linaje
dbt docs serve       # sirve un sitio web local con el linaje del pipeline
```

En `schema.yml` puedes agregar descripciones que aparecen en la documentación:

```yaml
models:
  - name: ventas_por_cliente
    description: "Ingresos totales y cantidad de órdenes por cliente, usado en el dashboard de ventas."
    columns:
      - name: total_gastado
        description: "Suma de todos los montos de órdenes del cliente, en USD."
```

## 5. dbt + BigQuery / Redshift

dbt se conecta a cualquier [[Relational Model & SQL Fundamentals|base SQL]] mediante adaptadores. Ejemplos:

| Plataforma | Adaptador | Particularidad |
|------------|-----------|----------------|
| **BigQuery** | `dbt-bigquery` | Modelos incrementales con `insert_overwrite` y particiones |
| **Redshift** | `dbt-redshift` | Soporte `DISTSTYLE`, `SORTKEY` en config |
| **Snowflake** | `dbt-snowflake` | `transient` tables, `merge` para incrementales |
| **Postgres** | `dbt-postgres` | Ideal para desarrollo local y CI rápido |

## 6. CI/CD con dbt

dbt se integra nativamente en pipelines de CI/CD:

```yaml
# .github/workflows/dbt.yml (ejemplo conceptual)
jobs:
  dbt-ci:
    steps:
      - run: dbt deps          # instala dependencias (paquetes)
      - run: dbt build         # run + test en orden
      - run: dbt docs generate # actualiza docs
```

En PRs típicamente ejecutas solo los modelos afectados con `dbt build --select state:modified+`, usando `dbt clone` para copiar datos de producción a un schema de desarrollo.

## 7. Common Mistakes

1. **No usar `ref()`**: hardcodear nombres de tablas → dbt no resuelve dependencias ni orden.
2. **Modelos demasiado grandes**: 500 líneas de SQL en un solo archivo → difícil de testear y mantener. Divide en modelos pequeños.
3. **No testear sources**: asumir que los datos raw son perfectos. Siempen agrega tests a `sources.yml`.
4. **Ignorar `is_incremental()`**: modelos incrementales que reprocesan todo cada vez → $$ innecesario.
5. **No versionar `profiles.yml`**: las credenciales de conexión no deben estar en el repo. Usa variables de entorno.

## Resumen

1. dbt convierte SQL en código: modelos, tests, documentación y dependencias resueltas automáticamente.
2. Los modelos son `SELECT` con `ref()` y `source()`; se materializan como view, table o incremental.
3. Los tests (genéricos y singulares) corren junto a los modelos con `dbt test`.
4. `dbt docs generate` produce documentación del linaje y descripciones de columnas.
5. Se conecta a BigQuery, Redshift, Snowflake, Postgres mediante adaptadores.
6. En CI/CD, `dbt build` ejecuta modelos + tests, y `dbt clone` replica datos para desarrollo.
7. Errores comunes: no usar `ref()`, modelos monolíticos, no testear fuentes, mala config incremental.

## Check Your Understanding

1. ¿Qué diferencia hay entre `ref()` y `source()` en dbt? <!-- `ref()` referencia otro modelo dbt (gestiona dependencias); `source()` referencia una tabla raw externa (definida en sources.yml) -->
2. ¿Cuándo usarías un modelo incremental en lugar de table o view? <!-- Cuando la tabla fuente es muy grande y solo cambian los datos recientes; el incremental procesa solo lo nuevo en cada ejecución -->
3. ¿Qué significa que un singular test falle (devuelva filas)? <!-- Que la condición de calidad se violó: hay datos que no cumplen la regla definida en la consulta SQL del test -->
4. ¿Por qué `profiles.yml` no debe versionarse en el repositorio? <!-- Contiene credenciales de base de datos - debe manejarse con variables de entorno o secretos del CI/CD -->

## Where to Go Next

- [[Data Warehousing & Lakehouse]]
- [[Data Pipelines & ETL]]
- [[Data Quality & Testing]]
- [[Advanced Querying]]
- [[CI-CD & GitOps]]
