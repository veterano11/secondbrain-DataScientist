---
tags:
  - data-engineering
  - data-quality
  - testing
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Llevas 3 meses con un pipeline de datos en producción y nadie revisa si los datos son correctos. Un día descubres que el campo `ingresos` tiene 40% de valores null desde hace 2 meses. El equipo de finanzas usó esos datos en reportes ejecutivos. Necesitas [[Data Quality & Testing|testing de calidad de datos]] para que esto no vuelva a pasar.

## 1. Dimensiones de calidad

La calidad de datos se mide en varias dimensiones:

| Dimensión | Pregunta | Ejemplo |
|-----------|----------|---------|
| **Completitud** | ¿Faltan valores? | `COUNT(*) vs COUNT(columna)` |
| **Unicidad** | ¿Hay duplicados? | `COUNT(DISTINCT id) vs COUNT(*)` |
| **Consistencia** | ¿Coinciden entre sistemas? | Monto total en OLTP vs en DW |
| **Validez** | ¿Cumplen el formato/reglas? | Email con @, fecha en rango |
| **Puntualidad** | ¿Llegaron a tiempo? | Datos de hoy disponibles antes de las 8 AM |
| **Integridad** | ¿Las relaciones son válidas? | Toda orden tiene un cliente existente |

## 2. Tests de datos en SQL

Antes de herramientas sofisticadas, el SQL es tu primera línea de defensa:

```sql
-- 1. Completitud: ¿cuántos nulls hay?
SELECT
    'ingresos' AS campo,
    COUNT(*) AS total_filas,
    SUM(CASE WHEN ingresos IS NULL THEN 1 ELSE 0 END) AS nulos,
    ROUND(100.0 * SUM(CASE WHEN ingresos IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_nulos
FROM ventas;

-- Salida esperada (cuando hay problemas):
-- campo    | total_filas | nulos  | pct_nulos
-- ingresos | 500000      | 200000 | 40.00

-- 2. Unicidad
SELECT id, COUNT(*) AS dups
FROM clientes
GROUP BY id
HAVING COUNT(*) > 1;

-- 3. Integridad referencial
SELECT v.cliente_id
FROM ventas v
LEFT JOIN clientes c ON v.cliente_id = c.id
WHERE c.id IS NULL;
```

## 3. Great Expectations

Great Expectations (GX) es una librería Python para definir, ejecutar y documentar expectativas de calidad:

```python
import great_expectations as gx

context = gx.get_context()
batch = context.sources.pandas_default.read_csv("ventas.csv")

# Definir expectativas
batch.expect_column_values_to_not_be_null("ingresos")
batch.expect_column_values_to_be_between("monto", 0, 1000000)
batch.expect_column_pair_values_to_be_equal("fecha_creacion", "fecha_actualizacion")

# Salida esperada:
# {
#   "success": True/False,
#   "statistics": {"evaluated_expectations": 3, "successful_expectations": 2, ...},
#   "results": [...]
# }
results = batch.validate()
```

GX genera **Data Docs** (HTML) que documentan la calidad de cada dataset, ideal para compartir con el equipo.

## 4. dbt tests

Si usas [[dbt & Data Transformation|dbt]], los tests son parte nativa del flujo. Puedes complementar los tests genéricos con `.yml` y añadir **singular tests** (consultas que deben dar 0 filas):

```sql
-- tests/ingresos_no_nulos.sql
-- Singular test: falla si hay ingresos null
SELECT *
FROM {{ ref('ventas') }}
WHERE ingresos IS NULL
  AND fecha >= '2026-01-01';
```

```yaml
# schema.yml
models:
  - name: ventas
    columns:
      - name: ingresos
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
      - name: cliente_id
        tests:
          - relationships:
              to: ref('clientes')
              field: id
```

## 5. Data contracts

Un **data contract** es un acuerdo formal entre productor y consumidor de datos que define:

- **Schema**: nombres, tipos, valores permitidos.
- **SLOs**: latencia máxima, completitud mínima (ej. "ingresos < 5% null").
- **Evolución**: solo cambios backward-compatible (nuevas columnas opcionales). Breaking changes requieren versión nueva.

Ejemplo de contrato en schema:

```yaml
contract:
  version: 1.2
  owners:
    producer: "equipo-pagos"
    consumer: "equipo-analytics"
  slos:
    - dimension: completitud
      column: ingresos
      threshold: 0.05  # máx 5% null
    - dimension: latencia
      max_hours: 4     # datos disponibles dentro de 4h
  schema:
    columns:
      - name: venta_id
        type: INT64
        nullable: false
      - name: ingresos
        type: DECIMAL(12,2)
        nullable: false
```

## 6. Monitoreo continuo

Los tests puntuales no bastan. Necesitas monitoreo continuo con:

1. **Alertas**: si un test de completitud falla, notificar a Slack/PagerDuty.
2. **Dashboards de calidad**: grafique el % de nulls y duplicados en el tiempo.
3. **SLOs**: objetivos medibles como "completitud de ingresos > 95% en las últimas 24h".

```sql
-- Consulta para dashboard de calidad: evolución de nulls
SELECT
    DATE_TRUNC('day', fecha) AS dia,
    COUNT(*) AS total,
    SUM(CASE WHEN ingresos IS NULL THEN 1 ELSE 0 END) AS nulos,
    ROUND(100.0 * SUM(CASE WHEN ingresos IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_nulos
FROM ventas
GROUP BY 1
ORDER BY 1;
```

## 7. Errores Comunes

1. **Tests sin alertas**: tests que fallan pero nadie se entera hasta 2 meses después.
2. **No testear fuentes**: asumir que los datos raw son perfectos. Siempre testea `sources` en dbt.
3. **Data contracts sin versionado**: cambiar tipos sin avisar rompe los consumidores.
4. **Solo tests de completitud**: ignorar unicidad, consistencia, validez y puntualidad.
5. **Thresholds absolutos**: "nunca debe haber nulls". Mejor un umbral realista (mayor al 5%) con alerta temprana al 3%.

## Resumen

1. Las dimensiones de calidad son: completitud, unicidad, consistencia, validez, puntualidad e integridad.
2. SQL simple puede detectar nulls, duplicados y relaciones rotas antes de que escalen.
3. Great Expectations permite definir expectativas en Python y generar documentación de calidad.
4. dbt tests integran calidad con el flujo de transformación (tests genéricos y singulares).
5. Data contracts formalizan acuerdos de schema, SLOs y evolución entre productor y consumidor.
6. El monitoreo continuo con alertas, dashboards y SLOs evita sorpresas de meses.
7. Errores comunes: tests sin alertas, no testear fuentes, contracts sin versionado, thresholds absolutos.

## Comprueba tu Conocimiento

1. ¿Qué dimensión de calidad se viola si un campo "email" tiene valores como "abc"? <!-- Validez: el formato del dato no cumple la regla de negocio (debe contener @) -->
2. ¿Cuál es la diferencia entre un test genérico y un singular test en dbt? <!-- Genérico: se define declarativamente en YML (not_null, unique). Singular: es una consulta SQL que debe devolver 0 filas -->
3. ¿Qué incluye un data contract además del schema? <!-- SLOs de calidad y latencia, dueños (productor/consumidor), reglas de evolución y versionado -->
4. ¿Por qué es importante tener alertas y no solo tests nocturnos? <!-- Porque si el test falla a las 2 AM, sin alerta el problema puede estar 2 meses sin detectarse -->

## ¿Dónde ir Siguente?

- [[Data Pipelines & ETL]]
- [[dbt & Data Transformation]]
- [[Workflow Orchestration (Airflow)]]
- [[Data Warehousing & Lakehouse]]
- [[Testing for Data Science]]
