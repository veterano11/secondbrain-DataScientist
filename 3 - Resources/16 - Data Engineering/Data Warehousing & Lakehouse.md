---
tags:
  - data-engineering
  - data-warehousing
  - lakehouse
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu equipo tiene datos en S3, Redshift, y tablas de Excel. Cada vez que alguien necesita un reporte mensual de ventas, alguien pasa días juntando datos de las tres fuentes, limpiándolos a mano y armando una consulta SQL que nadie documenta. Los reportes nunca son consistentes entre sí. Necesitas un [[Data Warehousing & Lakehouse|data warehouse]] bien diseñado para que los reportes sean rápidos, confiables y reproducibles.

## 1. OLTP vs OLAP

Los sistemas operacionales (OLTP) optimizan escrituras rápidas de transacciones individuales (una orden, un pago). Los analíticos (OLAP) optimizan lecturas masivas de datos agregados. La diferencia fundamental es el almacenamiento:

- **Row-oriented** (OLTP): cada fila se guarda contigua. Ideal para `SELECT *` de pocas filas.
- **Column-oriented** (OLAP): cada columna se guarda contigua. Ideal para `SELECT SUM(ventas) GROUP BY region` sobre millones de filas.

```sql
-- OLTP: busca una orden específica rápido
SELECT * FROM orders WHERE order_id = 42;

-- OLAP: agrega millones de órdenes rápido
-- Salida esperada:
-- region   | total_ventas
-- Noreste  | 15203400.50
-- Suroeste | 9872100.75
SELECT r.nombre_region, SUM(v.total) AS total_ventas
FROM ventas v
JOIN regiones r ON v.region_id = r.id
GROUP BY r.nombre_region;
```

## 2. Star Schema

El diseño más común en [[Data Warehousing & Lakehouse|data warehousing]] es el **esquema estrella**: una tabla de hechos (fact) rodeada de tablas de dimensiones.

| Elemento | Descripción | Ejemplo |
|----------|-------------|---------|
| **Fact table** | Medidas numéricas, claves foráneas a dimensiones | `ventas(venta_id, fecha_id, producto_id, cliente_id, monto)` |
| **Dimension table** | Atributos descriptivos, desnormalizados | `producto(producto_id, nombre, categoría, precio)` |

Ventajas: consultas simples con pocos JOINs, entendible por analistas de negocio.

## 3. Snowflake Schema

Es una normalización del star schema: las dimensiones se dividen en sub-dimensiones. Por ejemplo, `producto` se separa en `producto` y `categoría`. Reduce redundancia pero aumenta la cantidad de JOINs. Se usa cuando el mantenimiento de datos maestros es crítico y el rendimiento de consultas lo permite.

## 4. Partitioning y Bucketing

En [[Advanced Querying|sistemas analíticos]] modernos, dos técnicas mejoran drásticamente los scans:

| Técnica | Cómo funciona | Cuándo usarla |
|---------|---------------|---------------|
| **Partitioning** | Divide datos en directorios/particiones por una columna (ej. fecha) | Filtros frecuentes por esa columna |
| **Bucketing** | Hash de una columna en N archivos fijos | JOINs o agregaciones por esa columna |

```sql
-- Particionado por fecha en Hive/Spark SQL
CREATE TABLE ventas (
    producto_id INT,
    monto DECIMAL(10,2)
)
PARTITIONED BY (fecha DATE);

-- Salida esperada: los datos se organizan en carpetas como
-- ventas/fecha=2026-01-01/, ventas/fecha=2026-01-02/, ...
```

## 5. Data Lake vs Data Warehouse vs Lakehouse

| Arquitectura | Formato | Usuarios | Caso típico |
|-------------|---------|----------|-------------|
| **Data Lake** | Raw (JSON, Parquet, CSV) | Data Scientists, Data Engineers | Exploración, ML, datos sin procesar |
| **Data Warehouse** | Tablas SQL procesadas | Analistas, BI | Reportes, dashboards, KPI |
| **Lakehouse** | Tablas SQL sobre data lake (Delta/Iceberg) | Todos | Unifica lake + warehouse |

El **Lakehouse** (Databricks, Apache Iceberg, Delta Lake) es la tendencia actual porque da transacciones ACID y rendimiento SQL sobre objetos en S3, eliminando la necesidad de tener dos sistemas separados.

## 6. Herramientas modernas

- **Snowflake**: almacenamiento y cómputo separados, paga por uso, auto-scaling, zero-copy cloning.
- **BigQuery**: serverless, paga por datos escaneados, integración nativa con ML.
- **Redshift**: columnar en AWS, ideal si ya estás en el ecosistema.
- **Databricks**: Lakehouse sobre Spark, Delta Lake, notebooks colaborativos.

Todas soportan [[Query Optimization & Indexing|particionado, clustering y optimizaciones de consulta]].

## 7. Common Mistakes

1. **No particionar tablas grandes**: cada consulta escanea toda la tabla → lentitud y costo excesivo.
2. **JOINs entre fact tables**: genera productos cartesianos. Siempre pasa por dimensiones primero.
3. **Type-1 vs Type-2 SCDs**: confundir "sobrescribir" (Type-1) con "guardar historial" (Type-2). Si necesitas reportes históricos, usa Type-2.
4. **Modelo demasiado normalizado**: copias del star schema con 15 JOINs que nadie entiende.
5. **No documentar fuentes**: perder semanas rastreando de dónde viene `revenue_net`.

## Resumen

1. OLTP (row-oriented) para transacciones, OLAP (column-oriented) para análisis masivos.
2. Star schema: fact table + dimension tables desnormalizadas = consultas simples y rápidas.
3. Snowflake schema normaliza dimensiones, reduce redundancia pero aumenta JOINs.
4. Partitioning corta datos por columna; bucketing los distribuye uniformemente.
5. Data Lake = raw/barato; DW = procesado/rápido; Lakehouse = lo mejor de ambos.
6. Herramientas como Snowflake, BigQuery, Redshift y Databricks lideran el mercado.
7. Errores comunes: no particionar, joins entre facts, SCDs incorrectos, sobrenormalización.

## Check Your Understanding

1. ¿Cuál es la diferencia principal entre almacenamiento row-oriented y column-oriented? <!-- Row-oriented guarda filas contiguas (rápido para transacciones individuales); column-oriented guarda columnas contiguas (rápido para agregaciones masivas) -->
2. ¿Qué problema resuelve un star schema que un modelo 3NF no resuelve bien en analytics? <!-- Reduce la cantidad de JOINs necesarios para consultas analíticas típicas, haciendo las consultas más simples y rápidas -->
3. ¿Cuándo preferirías un Lakehouse sobre un Data Warehouse tradicional? <!-- Cuando necesitas datos sin procesar (ML, exploración) y tablas SQL analíticas en el mismo lugar, sin mover datos entre sistemas -->
4. ¿Por qué es peligroso hacer JOIN entre dos fact tables directamente? <!-- Porque puede generar productos cartesianos si no hay una dimensión que las relacione correctamente -->

## Where to Go Next

- [[Data Pipelines & ETL]]
- [[dbt & Data Transformation]]
- [[Advanced Querying]]
- [[Query Optimization & Indexing]]
- [[Data Quality & Testing]]
