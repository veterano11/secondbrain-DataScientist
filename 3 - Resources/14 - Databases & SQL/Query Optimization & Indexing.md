---
tags:
  - sql
  - databases
  - optimization
  - indexing
  - performance
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu query de reporte mensual — que antes volaba — ahora tarda **45 segundos**. La tabla `pedidos` creció a 10 millones de filas. El negocio necesita el dashboard en segundos, no en minutos.

El problema no es SQL en sí; es cómo se ejecuta. Tu query hace un **sequential scan** (lee cada fila de la tabla) cuando podría usar un **index scan** (salta directo a las filas relevantes). Aquí aprenderás a leer planes de ejecución, diseñar índices y reescribir queries para que sean órdenes de magnitud más rápidas.

Usaremos el mismo esquema de [[Relational Model & SQL Fundamentals|SQL Fundamentals]].

**Requisitos**: SQLite con datos generados (puedes usar `WITH RECURSIVE` para generar un millón de filas de prueba).

---

## 1. Cómo ejecuta SQL una query

Cuando mandas una query, el motor pasa por tres etapas:

```
SQL ──▶ Parser ──▶ Optimizer ──▶ Executor ──▶ Resultado
```

1. **Parser**: verifica sintaxis y convierte el texto en un árbol interno.
2. **Optimizer**: el cerebro. Genera múltiples planes de ejecución y estima el costo de cada uno (lecturas de disco, CPU, memoria). Elige el plan más barato.
3. **Executor**: ejecuta el plan paso a paso y devuelve filas.

El optimizer decide cosas como:

- ¿Usar un índice o leer toda la tabla?
- ¿Qué orden de JOIN es más barato?
- ¿Qué algoritmo de JOIN usar?

Entender el optimizer te ayuda a escribir queries que él pueda optimizar mejor. Este conocimiento es esencial para [[Data Engineering|pipelines en producción]].

---

## 2. EXPLAIN y EXPLAIN ANALYZE

`EXPLAIN` muestra el plan de ejecución sin ejecutar la query. `EXPLAIN QUERY PLAN` (SQLite) o `EXPLAIN ANALYZE` (PostgreSQL) la ejecutan y muestran tiempos reales.

```sql
-- ¿Cómo ejecuta SQLite esta query?
EXPLAIN QUERY PLAN
SELECT * FROM pedidos WHERE cliente_id = 1;
```

**Salida esperada:**
```
SEARCH pedidos USING INTEGER PRIMARY KEY (rowid=?)
```

Sin índice en `cliente_id`:

```sql
EXPLAIN QUERY PLAN
SELECT * FROM pedidos WHERE cliente_id = 1;
```

**Salida esperada:**
```
SCAN pedidos
```

`SCAN` = sequential scan (lee toda la tabla). Con 10M filas esto duele. `SEARCH` = usa índice (salta directo).

```sql
-- Query con JOIN
EXPLAIN QUERY PLAN
SELECT c.nombre, p.precio * pe.cantidad AS total
FROM clientes c
JOIN pedidos pe ON c.id = pe.cliente_id
JOIN productos p ON pe.producto_id = p.id
WHERE c.id = 1;
```

**Salida esperada:**
```
SEARCH c USING INTEGER PRIMARY KEY (rowid=?)
SCAN pe
SEARCH p USING INTEGER PRIMARY KEY (rowid=?)
```

Aquí `pedidos` se escanea completo. Necesitamos un [[Advanced Querying|índice]] en `cliente_id`.

---

## 3. Índices: CREATE INDEX, B-tree, compuestos

Un índice es una estructura separada que mapea valores de columna a ubicaciones de fila. Es como el índice de un libro: no lees página por página, vas directo.

### B-tree

El índice por defecto (y más común) es **B-tree**. Es un árbol balanceado que permite búsquedas, inserciones y borrados en O(log n).

### Índice simple

```sql
CREATE INDEX idx_pedidos_cliente ON pedidos(cliente_id);
```

Ahora el plan cambia:

```sql
EXPLAIN QUERY PLAN
SELECT * FROM pedidos WHERE cliente_id = 1;
```

**Salida esperada:**
```
SEARCH pedidos USING INDEX idx_pedidos_cliente (cliente_id=?)
```

### Índice compuesto

Útil cuando filtrar por varias columnas:

```sql
CREATE INDEX idx_pedidos_cliente_fecha ON pedidos(cliente_id, fecha);
```

```sql
-- Esta query usa el índice compuesto completo
SELECT * FROM pedidos
WHERE cliente_id = 1 AND fecha BETWEEN '2026-01-01' AND '2026-06-30';
```

**Regla de oro**: las columnas con mayor selectividad (más valores distintos) primero. [[Database Design & Normalization|El diseño del esquema]] influye directamente en qué índices tiene sentido crear.

---

## 4. Tipos de scan

| Scan | Descripción | Cuándo ocurre |
|------|-------------|---------------|
| **Sequential Scan** | Lee bloque por bloque toda la tabla | Tabla pequeña, o query sin filtro, o sin índice disponible |
| **Index Scan** | Busca en el índice, luego accede a la tabla por fila | Filtro con índice, pero necesitas columnas no indexadas |
| **Index-Only Scan** | Lee solo del índice, sin tocar la tabla | Todas las columnas del `SELECT` están en el índice |
| **Bitmap Scan** | Combina múltiples índices y luego accede a la tabla | Varios filtros con índices separados |

```sql
-- Index-Only Scan (si todas las columnas están en el índice)
CREATE INDEX idx_clientes_nombre_email ON clientes(nombre, email);

EXPLAIN QUERY PLAN
SELECT nombre, email FROM clientes WHERE nombre = 'Ana López';
```

La diferencia práctica entre estos scans es el tema central de [[Query Optimization & Indexing|optimización]] misma.

---

## 5. Optimización de JOINs

El optimizer elige entre tres algoritmos de JOIN. Conocerlos te ayuda a entender los planes de ejecución.

### Nested Loop JOIN

Para cada fila de la tabla A, busca coincidencias en la tabla B.

```
Para cada fila a en A:
    Para cada fila b en B:
        si a.key = b.key → emitir fila
```

**Costo**: O(n × m). Bueno para tablas pequeñas, o cuando una tabla tiene un índice y la otra es chica.

### Hash JOIN

Construye una tabla hash con la tabla más chica, luego escanea la grande buscando coincidencias.

```
Construir hash de B
Para cada fila a en A:
    buscar a.key en hash(B) → emitir si existe
```

**Costo**: O(n + m). Ideal para JOINs de muchas filas sin índice.

### Merge JOIN

Ordena ambas tablas por la clave de JOIN y luego las recorre en paralelo.

```
Ordenar A por key, ordenar B por key
Avanzar cursor en A y B sincronizadamente
```

**Costo**: O(n log n + m log m). Bueno cuando las tablas ya están ordenadas.

```sql
-- Fuerza a SQLite a mostrar algoritmo de JOIN
EXPLAIN
SELECT * FROM clientes c JOIN pedidos pe ON c.id = pe.cliente_id;
```

En SQLite siempre es **Nested Loop** (no tiene Hash ni Merge JOIN). PostgreSQL sí usa los tres. Si usas [[Python for Data Science|pandas]], los merge JOINs son el equivalente a `pd.merge()`.

---

## 6. Anti-patrones

Estos patrones matan el rendimiento aunque tengas índices:

### Función en WHERE

```sql
-- MAL: evita el índice (no es sargable)
SELECT * FROM pedidos WHERE strftime('%Y', fecha) = '2026';

-- BIEN: usa el índice
SELECT * FROM pedidos WHERE fecha >= '2026-01-01' AND fecha < '2027-01-01';
```

### SELECT *

```sql
-- MAL: trae columnas innecesarias, impide index-only scans
SELECT * FROM clientes WHERE email = 'ana@email.com';

-- BIEN: solo las columnas que necesitas
SELECT nombre, email FROM clientes WHERE email = 'ana@email.com';
```

### OR conditions

```sql
-- MAL: OR puede impedir uso de índice
SELECT * FROM pedidos WHERE cliente_id = 1 OR cliente_id = 2;

-- BIEN: IN usa índices correctamente
SELECT * FROM pedidos WHERE cliente_id IN (1, 2);
```

### Type casting implícito

```sql
-- MAL: SQLite convierte todas las filas a comparar
SELECT * FROM pedidos WHERE cliente_id = '1';  -- texto vs entero

-- BIEN: tipo correcto
SELECT * FROM pedidos WHERE cliente_id = 1;
```

Estos anti-patrones son errores comunes incluso entre desarrolladores experimentados. [[Code Quality|Buenas prácticas de código]] aplican también a SQL.

---

## 7. Common Mistakes

| Error | Explicación |
|-------|-------------|
| **Over-indexing** | Cada índice extra ralentiza `INSERT`/`UPDATE`/`DELETE`. No indexes columnas que no usas en filtros. |
| **Índice en columna de baja selectividad** | Indexar una columna booleana (true/false) apenas ayuda — el índice devuelve la mitad de la tabla. |
| **No revisar EXPLAIN** | Asumir que un índice se usa sin verificar el plan de ejecución. Siempre verifica con `EXPLAIN QUERY PLAN`. |
| **Falta de índices compuestos** | Dos índices separados en `(a)` y `(b)` no equivalen a un índice compuesto `(a, b)` para `WHERE a = ? AND b = ?`. |
| **No reindexar después de grandes cambios** | En SQLite: `REINDEX`; en PostgreSQL: `VACUUM`/`ANALYZE`. Los índices fragmentados pierden eficiencia. |
| **Asumir que todos los motores son iguales** | SQLite no usa Hash JOIN; PostgreSQL sí. Lo que funciona en uno puede ser pesimo en otro. |

---

## Resumen

1. Toda query pasa por Parser → Optimizer → Executor. El optimizer elige el plan más barato.
2. `EXPLAIN QUERY PLAN` revela si tu query hace sequential scan (lento) o index scan (rápido).
3. Los **índices B-tree** aceleran búsquedas por columna. Los **índices compuestos** ayudan con filtros múltiples.
4. Los tipos de scan son: sequential, index, index-only y bitmap scan.
5. Los JOINs usan algoritmos distintos: nested loop, hash join, merge join.
6. Los anti-patrones como funciones en `WHERE`, `SELECT *`, `OR` y type casting implícito rompen el uso de índices.
7. Over-indexing, no verificar EXPLAIN y asumir comportamiento idéntico entre motores son errores comunes.

---

## Check Your Understanding

1. ¿Qué comando usas para ver el plan de ejecución de una query en SQLite?
   <!-- EXPLAIN QUERY PLAN -->

2. ¿Por qué una función como `strftime('%Y', fecha)` en el WHERE rompe el uso de un índice?
   <!-- El índice almacena el valor original, no el resultado de la función. El motor no puede buscar por el valor transformado. -->

3. ¿Cuándo es mejor un índice compuesto `(a, b)` que dos índices separados en `a` y `b`?
   <!-- Cuando filtras por a AND b frecuentemente. Un índice compuesto permite index-only scan si SELECT solo pide a y b. -->

4. ¿Qué consecuencia tiene over-indexing?
   <!-- Las operaciones de escritura (INSERT/UPDATE/DELETE) se vuelven más lentas porque hay que mantener cada índice. -->

5. ¿Qué es un "sargable" predicate?
   <!-- Un predicado que puede usar un índice (Search ARGument ABLE). Ej: "columna = valor" es sargable; "funcion(columna) = valor" no. -->

---

## Where to Go Next

- [[Advanced Querying]] — window functions, CTEs, subqueries eficientes
- [[Relational Model & SQL Fundamentals]] — domina las bases antes de optimizar
- [[Database Design & Normalization]] — un buen diseño de schema reduce la necesidad de índices complejos
- [[Data Engineering]] — optimización en pipelines de datos a gran escala
- [[Code Quality]] — convenciones y buenas prácticas para SQL en producción
