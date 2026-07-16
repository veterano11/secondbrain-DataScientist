---
tags:
  - sql
  - databases
  - relational-model
  - fundamentals
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Eres analista en una tienda online. Tienes datos desperdigados en hojas de cálculo: clientes por un lado, productos por otro, pedidos en un tercer archivo. Cada vez que el negocio pregunta "¿cuántos pedidos hizo cada cliente este mes?" o "¿qué producto generó más ingresos?" pierdes una hora cruzando información manualmente.

El modelo relacional y SQL son la herramienta que necesitas. Con `CREATE TABLE` defines la estructura, con `INSERT` la llenas, y con `SELECT` + `JOIN` + `GROUP BY` respondes cualquier pregunta de negocio en segundos.

**Requisitos**: SQLite instalado. Verifica con:

```bash
$ sqlite3 --version
```

Si no aparece, instálalo con `brew install sqlite3` (macOS) o tu gestor de paquetes.

---

## 1. El modelo relacional en 5 minutos

El [[Relational Model & SQL Fundamentals|modelo relacional]] organiza datos en **tablas** (relaciones). Cada tabla tiene:

- **Filas** (tuplas / registros): representan una entidad (un cliente, un producto).
- **Columnas** (atributos): propiedades de esa entidad (nombre, precio, fecha).
- **Clave primaria** (PRIMARY KEY): columna o combinación que identifica **de forma única** cada fila.
- **Clave foránea** (FOREIGN KEY): columna que referencia la clave primaria de otra tabla — así se relacionan.

```
 clientes                          pedidos
┌──────────────┐                 ┌──────────────────┐
│ id  (PK)     │──┐             │ id  (PK)         │
│ nombre       │  └──FK────────>│ cliente_id (FK)  │
│ email        │                 │ producto_id (FK) │
└──────────────┘                 │ cantidad         │
                                 │ fecha            │
                                 └──────────────────┘
```

Las `FOREIGN KEY` son el pegamento que convierte tablas aisladas en un [[Database Design & Normalization|modelo relacional coherente]].

---

## 2. Tu primera base de datos

Vamos a crear el esquema de e-commerce. Abre SQLite y ejecuta:

```sql
CREATE TABLE clientes (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
);

CREATE TABLE productos (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    precio REAL NOT NULL
);

CREATE TABLE pedidos (
    id INTEGER PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    producto_id INTEGER NOT NULL,
    cantidad INTEGER NOT NULL,
    fecha TEXT NOT NULL,
    FOREIGN KEY (cliente_id) REFERENCES clientes(id),
    FOREIGN KEY (producto_id) REFERENCES productos(id)
);
```

Ahora inserta datos de ejemplo:

```sql
INSERT INTO clientes (nombre, email) VALUES
    ('Ana López', 'ana@email.com'),
    ('Luis Pérez', 'luis@email.com'),
    ('Carla Ruiz', 'carla@email.com');

INSERT INTO productos (nombre, precio) VALUES
    ('Laptop', 15000.00),
    ('Mouse', 350.00),
    ('Teclado', 650.00);

INSERT INTO pedidos (cliente_id, producto_id, cantidad, fecha) VALUES
    (1, 1, 1, '2026-06-01'),
    (1, 2, 2, '2026-06-05'),
    (2, 3, 1, '2026-06-10'),
    (2, 1, 1, '2026-06-15'),
    (3, 2, 3, '2026-06-20');
```

Puedes meter todo en un archivo `setup.sql` y ejecutarlo con:

```bash
$ sqlite3 tienda.db < setup.sql
```

Luego abre la base con `sqlite3 tienda.db` y empieza a consultar. El [[CLI & Productivity|uso eficiente de la terminal]] acelera mucho este flujo.

---

## 3. SELECT y WHERE

`SELECT` elige columnas, `WHERE` filtra filas. Ejemplos sobre nuestra tienda:

```sql
-- Todos los pedidos después del 10 de junio
SELECT * FROM pedidos WHERE fecha > '2026-06-10';
```

**Salida esperada:**
```
id|cliente_id|producto_id|cantidad|fecha
3|2|3|1|2026-06-10
4|2|1|1|2026-06-15
5|3|2|3|2026-06-20
```

```sql
-- Pedidos del cliente 1 con cantidad mayor a 1
SELECT id, cantidad, fecha
FROM pedidos
WHERE cliente_id = 1 AND cantidad > 1;
```

**Salida esperada:**
```
id|cantidad|fecha
2|2|2026-06-05
```

```sql
-- Clientes con nombre que empieza con 'A'
SELECT * FROM clientes WHERE nombre LIKE 'A%';
```

**Salida esperada:**
```
1|Ana López|ana@email.com
```

---

## 4. JOINs: unir tablas relacionadas

Los `JOIN` combinamos filas de dos tablas según una condición. [[Advanced Querying|Dominar los JOINs]] es la habilidad más importante en SQL.

### Comparación Visual de JOINs

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TIPOS DE JOIN                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INNER JOIN              LEFT JOIN               RIGHT JOIN         │
│  ┌─────┐ ┌─────┐       ┌─────┐ ┌─────┐        ┌─────┐ ┌─────┐   │
│  │█████│█│█████│       │█████│█│░░░░░│        │░░░░░│█│█████│   │
│  │█████│ │█████│       │█████│ │░░░░░│        │░░░░░│ │█████│   │
│  └─────┘ └─────┘       └─────┘ └─────┘        └─────┘ └─────┘   │
│  Solo coinciden         Todas las izq.          Todas las der.    │
│  en ambas               + coinciden             + coinciden       │
│                                                                     │
│  FULL OUTER JOIN        CROSS JOIN                                   │
│  ┌─────┐ ┌─────┐       ┌─────┐ ┌─────┐                            │
│  │█████│█│█████│       │█████│█│█████│                            │
│  │░░░░░│ │░░░░░│       │█████│ │█████│                            │
│  └─────┘ └─────┘       └─────┘ └─────┘                            │
│  Todas las filas         Combinación cartesiana                    │
│  de ambas tablas         (cada fila × cada fila)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### INNER JOIN — solo filas que coinciden en ambas tablas

```sql
SELECT c.nombre, p.nombre AS producto, pe.cantidad
FROM pedidos pe
INNER JOIN clientes c ON pe.cliente_id = c.id
INNER JOIN productos p ON pe.producto_id = p.id;
```

**Salida esperada:**
```
Ana López|Laptop|1
Ana López|Mouse|2
Luis Pérez|Teclado|1
Luis Pérez|Laptop|1
Carla Ruiz|Mouse|3
```

### LEFT JOIN — todas las filas de la izquierda, aunque no tengan match

```sql
SELECT c.nombre, COUNT(pe.id) AS pedidos
FROM clientes c
LEFT JOIN pedidos pe ON c.id = pe.cliente_id
GROUP BY c.id;
```

**Salida esperada:**
```
Ana López|2
Luis Pérez|2
Carla Ruiz|1
```

### RIGHT JOIN — inverso del LEFT (no soportado en SQLite; se simula invirtiendo tablas)

```sql
-- Equivalente a RIGHT JOIN en PostgreSQL:
SELECT c.nombre, pe.id AS pedido_id
FROM pedidos pe
RIGHT JOIN clientes c ON pe.cliente_id = c.id;
```

> Nota: En SQLite no hay `RIGHT JOIN` ni `FULL JOIN`. Usa `LEFT JOIN` con las tablas intercambiadas.

---

## 5. Agregaciones: GROUP BY, COUNT, SUM, AVG, HAVING

Las funciones de agregación resumen muchas filas en una. Con `GROUP BY` defines los grupos.

```sql
-- Gasto total por cliente
SELECT c.nombre, SUM(p.precio * pe.cantidad) AS gasto_total
FROM clientes c
JOIN pedidos pe ON c.id = pe.cliente_id
JOIN productos p ON pe.producto_id = p.id
GROUP BY c.id;
```

**Salida esperada:**
```
Ana López|15700.0
Luis Pérez|15650.0
Carla Ruiz|1050.0
```

```sql
-- Productos con más de 1 unidad vendida en total
SELECT p.nombre, SUM(pe.cantidad) AS unidades
FROM productos p
JOIN pedidos pe ON p.id = pe.producto_id
GROUP BY p.id
HAVING SUM(pe.cantidad) > 1;
```

**Salida esperada:**
```
Mouse|5
Laptop|2
```

`HAVING` es como `WHERE` pero se aplica **después** de la agregación. La diferencia es clave en [[Query Optimization & Indexing|queries optimizadas]].

---

## 6. ORDER BY y LIMIT

Ordenas resultados con `ORDER BY` y limitas filas con `LIMIT`.

```sql
-- Top 2 productos más caros
SELECT nombre, precio FROM productos ORDER BY precio DESC LIMIT 2;
```

**Salida esperada:**
```
Laptop|15000.0
Teclado|650.0
```

```sql
-- Cliente con mayor gasto (usando subconsulta o simplemente ordenando)
SELECT c.nombre, SUM(p.precio * pe.cantidad) AS total
FROM clientes c
JOIN pedidos pe ON c.id = pe.cliente_id
JOIN productos p ON pe.producto_id = p.id
GROUP BY c.id
ORDER BY total DESC
LIMIT 1;
```

**Salida esperada:**
```
Ana López|15700.0
```

---

## 7. Common Mistakes

| Error | Explicación |
|-------|-------------|
| `WHERE nombre = NULL` | `NULL` se compara con `IS NULL`, no con `=`. Correcto: `WHERE nombre IS NULL`. |
| `SELECT *` en producción | Devuelve columnas innecesarias, rompe si cambia el schema. Prefiere columnas explícitas. |
| Olvidar la condición de JOIN | Si haces `FROM a JOIN b` sin `ON`, obtienes producto cartesiano (cada fila de a × cada fila de b). |
| Usar `HAVING` donde va `WHERE` | `WHERE` filtra antes de agrupar; `HAVING` filtra después. Usa `WHERE` siempre que puedas — es más rápido. |
| No definir PRIMARY KEY | SQLite permite tablas sin PK, pero pierdes integridad y [[Query Optimization & Indexing|rendimiento en joins]]. |

---

## Resumen

1. El modelo relacional organiza datos en tablas con claves primarias y foráneas.
2. `CREATE TABLE` + `INSERT` construyen el esquema y los datos.
3. `SELECT` + `WHERE` filtran filas por condiciones.
4. Los `JOIN`s (INNER, LEFT) combinan tablas relacionadas.
5. `GROUP BY` + funciones de agregación (`SUM`, `COUNT`, `AVG`) resumen datos.
6. `ORDER BY` + `LIMIT` ordenan y acotan resultados.
7. Los errores comunes tienen soluciones conocidas — revisa siempre `NULL` y condiciones de JOIN.

---

## Check Your Understanding

1. ¿Qué tipo de JOIN usarías para listar todos los clientes, incluso los que nunca compraron?
   <!-- LEFT JOIN (clientes LEFT JOIN pedidos) -->

2. ¿Cuál es la diferencia entre `WHERE` y `HAVING`?
   <!-- WHERE filtra antes de agrupar; HAVING filtra grupos después de la agregación -->

3. ¿Cómo cuentas cuántos pedidos hizo cada cliente?
   <!-- SELECT cliente_id, COUNT(*) FROM pedidos GROUP BY cliente_id -->

4. ¿Por qué `SELECT *` se considera mala práctica en producción?
   <!-- Porque devuelve columnas innecesarias y el resultado cambia si la tabla se modifica -->

5. ¿Qué ocurre si haces un JOIN sin cláusula ON?
   <!-- Producto cartesiano: cada fila de la tabla A se combina con todas las filas de la tabla B -->

---

## Where to Go Next

- [[Advanced Querying]] — subqueries, CTEs, window functions
- [[Python for Data Science]] — usa SQL desde pandas y Jupyter
- [[CLI & Productivity]] — domina la terminal para trabajar con bases de datos
- [[Data Engineering]] — pipelines, ETL, bases de datos en producción
- [[Database Design & Normalization]] — cómo diseñar esquemas que no apesten
