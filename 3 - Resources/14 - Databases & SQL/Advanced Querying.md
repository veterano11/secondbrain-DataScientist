---
tags:
  - sql
  - databases
  - advanced-querying
  - cte
  - window-functions
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tus queries básicas con `JOIN` y `GROUP BY` funcionan bien. Pero llegan preguntas más difíciles:

- "Tráeme el ranking de clientes por gasto total **cada mes**."
- "¿Cuál fue la diferencia entre el pedido de este mes y el anterior para cada cliente?"
- "¿Cuáles son los 3 productos más vendidos de cada categoría este trimestre?"

Con `GROUP BY` solo no puedes: necesitas **subqueries**, **CTEs** (Common Table Expressions) y **window functions**. Estas herramientas te permiten hacer cálculos entre filas sin perder detalle.

Usaremos el mismo esquema de la nota [[Relational Model & SQL Fundamentals]], así que asegúrate de tener `tienda.db` lista.

**Requisitos**: SQLite 3.25+ (soporta window functions). Verifica:

```bash
$ sqlite3 --version
```

---

## 1. Subqueries

Una subquery es un `SELECT` dentro de otro `SELECT`. Puede ir en `WHERE`, `FROM`, o `SELECT`.

### Subquery en WHERE

```sql
-- Clientes que gastaron más que el promedio general
SELECT c.nombre, SUM(p.precio * pe.cantidad) AS gasto
FROM clientes c
JOIN pedidos pe ON c.id = pe.cliente_id
JOIN productos p ON pe.producto_id = p.id
GROUP BY c.id
HAVING gasto > (SELECT AVG(total) FROM (
    SELECT SUM(p2.precio * pe2.cantidad) AS total
    FROM pedidos pe2
    JOIN productos p2 ON pe2.producto_id = p2.id
    GROUP BY pe2.cliente_id
));
```

**Salida esperada:**
```
Ana López|15700.0
Luis Pérez|15650.0
```

### Subquery en FROM (derived table)

```sql
-- Gasto mensual por cliente usando una tabla derivada
SELECT c.nombre, mensual.mes, mensual.gasto
FROM clientes c
JOIN (
    SELECT
        pe.cliente_id,
        strftime('%Y-%m', pe.fecha) AS mes,
        SUM(p.precio * pe.cantidad) AS gasto
    FROM pedidos pe
    JOIN productos p ON pe.producto_id = p.id
    GROUP BY pe.cliente_id, mes
) mensual ON c.id = mensual.cliente_id
ORDER BY c.nombre, mensual.mes;
```

**Salida esperada:**
```
Ana López|2026-06|15700.0
Carla Ruiz|2026-06|1050.0
Luis Pérez|2026-06|15650.0
```

### EXISTS vs IN

`EXISTS` suele ser más eficiente que `IN` con subqueries grandes:

```sql
-- Clientes que han comprado al menos una vez (con EXISTS)
SELECT * FROM clientes c
WHERE EXISTS (
    SELECT 1 FROM pedidos pe WHERE pe.cliente_id = c.id
);
```

```sql
-- Clientes que NO han comprado nunca (con NOT EXISTS)
SELECT * FROM clientes c
WHERE NOT EXISTS (
    SELECT 1 FROM pedidos pe WHERE pe.cliente_id = c.id
);
```

> 💡 `EXISTS` corta en cuanto encuentra la primera coincidencia. `IN` materializa toda la subquery primero — diferencia importante en [[Query Optimization & Indexing|rendimiento]].

---

## 2. Common Table Expressions (CTEs)

Un CTE (cláusula `WITH`) es como una "variable temporal" para tu query. Mejora la [[Code Quality|legibilidad]] y permite reutilizar lógica.

```sql
-- CTE básico: gasto por cliente
WITH gasto_cliente AS (
    SELECT
        pe.cliente_id,
        SUM(p.precio * pe.cantidad) AS total
    FROM pedidos pe
    JOIN productos p ON pe.producto_id = p.id
    GROUP BY pe.cliente_id
)
SELECT c.nombre, gc.total
FROM clientes c
JOIN gasto_cliente gc ON c.id = gc.cliente_id
ORDER BY gc.total DESC;
```

**Salida esperada:**
```
Ana López|15700.0
Luis Pérez|15650.0
Carla Ruiz|1050.0
```

### CTEs múltiples

```sql
WITH
gasto_cliente AS (
    SELECT pe.cliente_id, SUM(p.precio * pe.cantidad) AS total
    FROM pedidos pe JOIN productos p ON pe.producto_id = p.id
    GROUP BY pe.cliente_id
),
promedio_general AS (
    SELECT AVG(total) AS promedio FROM gasto_cliente
)
SELECT c.nombre, gc.total, pg.promedio
FROM clientes c
JOIN gasto_cliente gc ON c.id = gc.cliente_id
CROSS JOIN promedio_general pg
WHERE gc.total > pg.promedio;
```

**Salida esperada:**
```
Ana López|15700.0|10733.33
Luis Pérez|15650.0|10733.33
```

Los CTEs son la antesala perfecta para [[Time Series Fundamentals|cálculos temporales avanzados]].

---

## 3. Window Functions — ROW_NUMBER, RANK, DENSE_RANK

Las window functions hacen cálculos **entre filas de un grupo** sin colapsarlas como `GROUP BY`. Cada fila conserva su identidad.

Necesitan una cláusula `OVER` que define la "ventana". `PARTITION BY` divide en grupos; `ORDER BY` ordena dentro de cada grupo.

### ROW_NUMBER

```sql
-- Ranking de clientes por gasto (sin empates posibles)
SELECT
    c.nombre,
    SUM(p.precio * pe.cantidad) AS gasto,
    ROW_NUMBER() OVER (ORDER BY SUM(p.precio * pe.cantidad) DESC) AS ranking
FROM clientes c
JOIN pedidos pe ON c.id = pe.cliente_id
JOIN productos p ON pe.producto_id = p.id
GROUP BY c.id;
```

**Salida esperada:**
```
Ana López|15700.0|1
Luis Pérez|15650.0|2
Carla Ruiz|1050.0|3
```

### RANK y DENSE_RANK

```sql
-- RANK salta números si hay empates; DENSE_RANK no
SELECT
    p.nombre,
    SUM(pe.cantidad) AS vendidos,
    RANK()       OVER (ORDER BY SUM(pe.cantidad) DESC) AS rank_rank,
    DENSE_RANK() OVER (ORDER BY SUM(pe.cantidad) DESC) AS dense_rank
FROM productos p
JOIN pedidos pe ON p.id = pe.producto_id
GROUP BY p.id;
```

**Salida esperada:**
```
Mouse|5|1|1
Laptop|2|2|2
Teclado|1|3|3
```

---

## 4. Window Functions — LAG, LEAD, SUM OVER

### LAG — valor de la fila anterior

```sql
-- Diferencia de gasto entre pedidos consecutivos de cada cliente
WITH pedidos_ordenados AS (
    SELECT
        pe.cliente_id,
        pe.fecha,
        p.precio * pe.cantidad AS total
    FROM pedidos pe
    JOIN productos p ON pe.producto_id = p.id
)
SELECT
    cliente_id,
    fecha,
    total,
    LAG(total) OVER (PARTITION BY cliente_id ORDER BY fecha) AS pedido_anterior,
    total - LAG(total) OVER (PARTITION BY cliente_id ORDER BY fecha) AS diferencia
FROM pedidos_ordenados;
```

**Salida esperada:**
```
1|2026-06-01|15000.0||
1|2026-06-05|700.0|15000.0|-14300.0
2|2026-06-10|650.0||
2|2026-06-15|15000.0|650.0|14350.0
3|2026-06-20|1050.0||
```

### SUM OVER — total acumulado (rolling total)

```sql
-- Gasto acumulado por cliente en el tiempo
SELECT
    pe.cliente_id,
    pe.fecha,
    p.precio * pe.cantidad AS total,
    SUM(p.precio * pe.cantidad) OVER (
        PARTITION BY pe.cliente_id
        ORDER BY pe.fecha
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS acumulado
FROM pedidos pe
JOIN productos p ON pe.producto_id = p.id
ORDER BY pe.cliente_id, pe.fecha;
```

**Salida esperada:**
```
1|2026-06-01|15000.0|15000.0
1|2026-06-05|700.0|15700.0
2|2026-06-10|650.0|650.0
2|2026-06-15|15000.0|15650.0
3|2026-06-20|1050.0|1050.0
```

El rolling total con `SUM OVER` es una técnica que también aparece en [[Python for Data Science|cálculos con pandas]] (`.cumsum()`).

---

## 5. CTEs + Window Functions: top 3 por categoría cada mes

Combinando ambas herramientas resolvemos el escenario inicial:

```sql
WITH ventas_mensuales AS (
    SELECT
        strftime('%Y-%m', pe.fecha) AS mes,
        p.nombre AS producto,
        SUM(pe.cantidad) AS unidades,
        DENSE_RANK() OVER (
            PARTITION BY strftime('%Y-%m', pe.fecha)
            ORDER BY SUM(pe.cantidad) DESC
        ) AS rank
    FROM pedidos pe
    JOIN productos p ON pe.producto_id = p.id
    GROUP BY mes, producto
)
SELECT mes, producto, unidades, rank
FROM ventas_mensuales
WHERE rank <= 3
ORDER BY mes, rank;
```

**Salida esperada (con datos pequeños — amplía la DB para ver el poder real):**
```
2026-06|Mouse|5|1
2026-06|Laptop|2|2
2026-06|Teclado|1|3
```

Este patrón (CTE + window function + filtro final) es estándar en [[Data Engineering|pipelines de datos]] y reportes de negocio.

---

## 6. CASE WHEN — lógica condicional en SQL

`CASE WHEN` permite categorizar datos dentro de una query, como un `if/else`.

```sql
-- Segmentar clientes por su gasto total
WITH gasto AS (
    SELECT c.id, c.nombre,
           COALESCE(SUM(p.precio * pe.cantidad), 0) AS total
    FROM clientes c
    LEFT JOIN pedidos pe ON c.id = pe.cliente_id
    LEFT JOIN productos p ON pe.producto_id = p.id
    GROUP BY c.id
)
SELECT
    nombre,
    total,
    CASE
        WHEN total >= 10000 THEN 'Alto'
        WHEN total >= 5000 THEN 'Medio'
        WHEN total > 0 THEN 'Bajo'
        ELSE 'Sin compras'
    END AS segmento
FROM gasto
ORDER BY total DESC;
```

**Salida esperada:**
```
Ana López|15700.0|Alto
Luis Pérez|15650.0|Alto
Carla Ruiz|1050.0|Bajo
```

Útil para [[Database Design & Normalization|vistas materializadas]] y tablas de reportes.

---

## 7. Common Mistakes

| Error | Explicación |
|-------|-------------|
| `ORDER BY` incorrecto en window function | Sin `ORDER BY` en `OVER`, las filas no tienen orden definido y funciones como `LAG` o `ROW_NUMBER` dan resultados inconsistentes. |
| Mezclar `GROUP BY` y window functions sin cuidado | Si haces `GROUP BY` y luego una window function, la ventana opera sobre los grupos, no sobre las filas originales. Entiende qué estás agregando y qué estás ventaneando. |
| Olvidar `ROWS BETWEEN` en ventanas acumulativas | Por defecto `ORDER BY` en window function usa `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, pero en algunos motores el comportamiento cambia. Sé explícito: `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. |
| Subquery retorna más de una fila en `=` | Si usas `= (subquery)`, la subquery debe devolver exactamente una fila. Para múltiples filas usa `IN` o `EXISTS`. |
| CTEs sin necesidad real | Para queries simples un CTE añade complejidad innecesaria. Úsalos cuando mejoran legibilidad o evitan repetir lógica. |

---

## Resumen

1. Las **subqueries** permiten usar el resultado de un `SELECT` dentro de otro — en `WHERE`, `FROM` o `SELECT`.
2. Los **CTEs** (`WITH`) nombran subqueries para reutilizarlas y mejorar legibilidad.
3. `ROW_NUMBER`, `RANK` y `DENSE_RANK` asignan números de ranking dentro de grupos sin colapsar filas.
4. `LAG` y `LEAD` acceden a filas anteriores/posteriores; `SUM OVER` con `ROWS BETWEEN` hace totales acumulados.
5. CTEs + window functions resuelven problemas reales como "top N por grupo y período".
6. `CASE WHEN` añade lógica condicional directamente en SQL.

---

## Check Your Understanding

1. ¿Qué diferencia hay entre `RANK()` y `DENSE_RANK()`?
   <!-- RANK salta números en empates (1, 1, 3); DENSE_RANK no (1, 1, 2). -->

2. ¿Qué hace `LAG(total) OVER (PARTITION BY cliente_id ORDER BY fecha)`?
   <!-- Devuelve el valor de "total" de la fila anterior dentro del mismo cliente, ordenado por fecha. -->

3. ¿Por qué `EXISTS` suele ser más eficiente que `IN`?
   <!-- EXISTS corta en la primera coincidencia; IN materializa toda la subquery antes de comparar. -->

4. ¿Cuándo usarías un CTE en lugar de una subquery anidada?
   <!-- Cuando la misma subquery se usa varias veces, o para mejorar legibilidad en queries complejas. -->

5. ¿Qué devuelve `SUM(x) OVER (ORDER BY y)` sin `PARTITION BY`?
   <!-- Un total acumulado global (sin división en grupos) ordenado por y. -->

---

## Where to Go Next

- [[Relational Model & SQL Fundamentals]] — si necesitas repasar JOINs y agregaciones básicas
- [[Query Optimization & Indexing]] — cómo hacer que estas queries vuelen en tablas grandes
- [[Database Design & Normalization]] — modelos bien diseñados = queries más simples
- [[Python for Data Science]] — lleva estas técnicas a pandas y DataFrames
- [[Time Series Fundamentals]] — cálculos temporales avanzados con ventanas
