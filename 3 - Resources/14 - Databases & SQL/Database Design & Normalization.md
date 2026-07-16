---
tags:
  - sql
  - databases
  - normalization
  - database-design
  - schema
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Te asignan a un proyecto nuevo: el e-commerce "TechStore" necesita base de datos desde cero. El CTO te dice "saca los requerimientos con el equipo de producto y diseña el esquema". Hablas con ellos y descubres que manejan clientes, productos, pedidos, proveedores y categorías. Como no conoces [[Relational Model & SQL Fundamentals|los fundamentos del modelo relacional]] aún, tu primer instinto es crear una tabla gigante con todo. Spoiler: es una pésima idea.

**Requisitos**: SQLite instalado y haber leído [[Relational Model & SQL Fundamentals]].

```bash
$ sqlite3 --version
```

---

## 1. Primer intento — el desastre de una sola tabla

### Visualización del Problema

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EL PROBLEMA: UNA SOLA TABLA                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ TABLA "todo" (MAL DISEÑO)                                   │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ id │ cliente_nombre │ cliente_email │ producto_nombre │ ... │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ 1  │ Ana López      │ ana@email.com │ Laptop          │ ... │   │
│  │ 2  │ Ana López      │ ana@email.com │ Mouse           │ ... │   │
│  │ 3  │ Luis Pérez     │ luis@email.com│ Teclado         │ ... │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  PROBLEMAS:                                                        │
│  ✗ Redundancia: "Ana López" aparece en cada fila                   │
│  ✗ Anomalía de actualización: cambiar dirección = N filas          │
│  ✗ Anomalía de borrado: borrar pedido = perder producto            │
│  ✗ Anomalía de inserción: no se puede registrar proveedor          │
│                                                                     │
│  SOLUCIÓN: Normalizar en múltiples tablas                          │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │  clientes   │  │  productos  │  │  pedidos    │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Abres SQLite y escribes esto:

```sql
CREATE TABLE todo (
    id INTEGER PRIMARY KEY,
    cliente_nombre TEXT,
    cliente_email TEXT,
    cliente_direccion TEXT,
    producto_nombre TEXT,
    producto_precio REAL,
    producto_categoria TEXT,
    pedido_fecha TEXT,
    pedido_cantidad INTEGER,
    proveedor_nombre TEXT,
    proveedor_telefono TEXT
);
```

Parece simple, pero insertas datos y empiezan los problemas:

```sql
INSERT INTO todo VALUES
    (1, 'Ana López', 'ana@email.com', 'Av. Siempre Viva 123',
     'Laptop', 15000, 'Electrónica', '2026-06-01', 1, 'TechSupply', '555-0100'),
    (2, 'Ana López', 'ana@email.com', 'Av. Siempre Viva 123',
     'Mouse', 350, 'Periféricos', '2026-06-05', 2, 'MouseWorld', '555-0200'),
    (3, 'Luis Pérez', 'luis@email.com', 'Calle Falsa 456',
     'Teclado', 650, 'Periféricos', '2026-06-10', 1, 'TechSupply', '555-0100');
```

**Problemas evidentes:**

- **Redundancia**: "Ana López" aparece con su email y dirección en cada fila.
- **Anomalía de actualización**: si Ana cambia de dirección, hay que actualizar N filas.
- **Anomalía de borrado**: si borramos el único pedido de un producto, perdemos los datos del producto y proveedor.
- **Anomalía de inserción**: no podemos registrar un proveedor hasta que venda algo.
- **Columnas repetitivas**: si un pedido tiene 5 productos, ¿agregamos `producto2_nombre`, `producto2_precio`?

Este diseño viola las formas normales. La [[Data Engineering|ingeniería de datos]] en serio empieza con un esquema bien normalizado.

---

## 2. First Normal Form (1NF)

### Diagrama de Normalización

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESO DE NORMALIZACIÓN                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ANTES (1 tabla):                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ todo: id, cliente_nombre, cliente_email, producto_nombre,   │   │
│  │       producto_precio, pedido_fecha, pedido_cantidad, ...   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼ 1NF: valores atómicos               │
│  DESPUÉS 1NF:                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ clientes: id, nombre, email, dirección                     │   │
│  │ productos: id, nombre, precio                              │   │
│  │ pedidos: id, cliente_id, producto_id, cantidad, fecha      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼ 2NF: dependencia parcial            │
│  DESPUÉS 2NF:                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ clientes: id, nombre, email, dirección                     │   │
│  │ productos: id, nombre, precio                              │   │
│  │ pedidos: id, cliente_id, fecha                             │   │
│  │ pedido_items: pedido_id, producto_id, cantidad             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼ 3NF: dependencia transitiva        │
│  DESPUÉS 3NF:                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ clientes: id, nombre, email, dirección                     │   │
│  │ productos: id, nombre, precio, categoria_id                │   │
│  │ categorias: id, nombre                                     │   │
│  │ pedidos: id, cliente_id, fecha                             │   │
│  │ pedido_items: pedido_id, producto_id, cantidad             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Una tabla está en **1NF** si:

1. Cada celda contiene un **valor atómico** (un solo valor, no listas ni conjuntos).
2. No hay **columnas repetitivas** (no `producto1`, `producto2`, `producto3`).
3. Cada fila es **única** (tiene una PRIMARY KEY).

Nuestra tabla `todo` falla porque:
- Un pedido podría tener varios productos → necesitaríamos columnas repetitivas o meter listas separadas por coma (violando atomicidad).
- Los datos del cliente se repiten → no hay separación de entidades.

**Solución**: separar en tablas por entidad (cliente, producto, pedido).

```sql
CREATE TABLE clientes (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    direccion TEXT NOT NULL
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

Esto ya está en **1NF**: cada columna es atómica, sin grupos repetitivos, y cada fila tiene identidad única.

**Salida esperada** — inserción limpia sin redundancia:

```sql
INSERT INTO clientes (nombre, email, direccion) VALUES
    ('Ana López', 'ana@email.com', 'Av. Siempre Viva 123'),
    ('Luis Pérez', 'luis@email.com', 'Calle Falsa 456');

INSERT INTO productos (nombre, precio) VALUES
    ('Laptop', 15000),
    ('Mouse', 350),
    ('Teclado', 650);

INSERT INTO pedidos (cliente_id, producto_id, cantidad, fecha) VALUES
    (1, 1, 1, '2026-06-01'),
    (1, 2, 2, '2026-06-05'),
    (2, 3, 1, '2026-06-10');
```

```
-- Sin duplicados de cliente, sin duplicados de producto
-- Cada tabla guarda solo lo que le corresponde
```

---

## 3. Second Normal Form (2NF)

Una tabla está en **2NF** si:

1. Está en **1NF**.
2. **Todos los atributos no clave dependen de la clave primaria completa** (no solo de una parte).

Esto solo aplica cuando la PK es **compuesta**. Mira esta tabla mal diseñada:

```sql
-- MAL: PK compuesta (cliente_id, producto_id)
CREATE TABLE pedidos_mal (
    cliente_id INTEGER,
    producto_id INTEGER,
    cliente_nombre TEXT,      -- Depende solo de cliente_id, no de la PK completa
    producto_nombre TEXT,      -- Depende solo de producto_id, no de la PK completa
    cantidad INTEGER,
    fecha TEXT,
    PRIMARY KEY (cliente_id, producto_id)
);
```

Aquí `cliente_nombre` depende **parcialmente** de `cliente_id` (parte de la PK). `producto_nombre` depende solo de `producto_id`. Esto genera redundancia: si el mismo cliente compra 10 productos, su nombre aparece 10 veces.

**Solución**: dividir en tablas separadas — cada atributo no clave debe depender de **toda** la PK. Como ya separamos cliente, producto y pedido en la sección anterior, nuestro esquema ya cumple 2NF: cada atributo depende de la clave completa de su tabla.

```sql
-- clientes:  id (PK) → nombre, email, direccion  ✅ depende de toda la PK
-- productos: id (PK) → nombre, precio             ✅ depende de toda la PK
-- pedidos:   id (PK) → cliente_id, producto_id, fecha, cantidad ✅ (PK simple)
```

**Salida esperada** — sin dependencias parciales:

```sql
-- Podemos cambiar el nombre de un cliente en UN solo lugar
UPDATE clientes SET nombre = 'Ana López García' WHERE id = 1;
```

```
-- Una fila afectada, no importa cuántos pedidos tenga Ana
```

---

## 4. Third Normal Form (3NF)

Una tabla está en **3NF** si:

1. Está en **2NF**.
2. **No hay dependencias transitivas**: un atributo no clave no debe depender de otro atributo no clave.

Ejemplo de dependencia transitiva:

```sql
-- MAL: dependencia transitiva
CREATE TABLE pedidos (
    id INTEGER PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    cliente_categoria TEXT,       -- Depende de cliente_id, no de pedido.id
    producto_id INTEGER NOT NULL,
    producto_proveedor TEXT,      -- Depende de producto_id, no de pedido.id
    cantidad INTEGER,
    fecha TEXT
);
```

`cliente_categoria` depende de `cliente_id`, que no es clave aquí. `producto_proveedor` depende de `producto_id`. Para eliminar transitividades, cada tabla debe modelar una **única entidad** y sus atributos deben depender **solo de su PK**.

Nuestro esquema hasta ahora está bien, pero agreguemos `proveedores` y `categorias` para ilustrar:

```sql
-- Tabla independiente para proveedores (elimina dependencia transitiva)
CREATE TABLE proveedores (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    telefono TEXT NOT NULL,
    email TEXT
);

-- Ahora productos tiene FK a proveedor, no guardamos datos del proveedor aquí
CREATE TABLE productos (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    precio REAL NOT NULL,
    categoria_id INTEGER NOT NULL,
    proveedor_id INTEGER NOT NULL,
    FOREIGN KEY (categoria_id) REFERENCES categorias(id),
    FOREIGN KEY (proveedor_id) REFERENCES proveedores(id)
);

CREATE TABLE categorias (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL UNIQUE
);
```

**Salida esperada** — el esquema refleja el mundo real sin transitividades:

```sql
SELECT p.nombre AS producto, cat.nombre AS categoria, prov.nombre AS proveedor
FROM productos p
JOIN categorias cat ON p.categoria_id = cat.id
JOIN proveedores prov ON p.proveedor_id = prov.id;
```

```
producto|categoria|proveedor
Laptop|Electrónica|TechSupply
Mouse|Periféricos|MouseWorld
Teclado|Periféricos|TechSupply
```

---

## 5. Diseño final normalizado

Juntando todo, este es el esquema 3NF completo para TechStore:

```sql
-- 1. Clientes
CREATE TABLE clientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    direccion TEXT NOT NULL,
    telefono TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 2. Categorías
CREATE TABLE categorias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL UNIQUE,
    descripcion TEXT
);

-- 3. Proveedores
CREATE TABLE proveedores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    contacto_nombre TEXT,
    telefono TEXT NOT NULL,
    email TEXT,
    direccion TEXT
);

-- 4. Productos
CREATE TABLE productos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    precio REAL NOT NULL CHECK (precio > 0),
    stock INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
    categoria_id INTEGER NOT NULL,
    proveedor_id INTEGER NOT NULL,
    FOREIGN KEY (categoria_id) REFERENCES categorias(id),
    FOREIGN KEY (proveedor_id) REFERENCES proveedores(id)
);

-- 5. Pedidos (cabecera)
CREATE TABLE pedidos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER NOT NULL,
    fecha TEXT NOT NULL DEFAULT (datetime('now')),
    total REAL NOT NULL CHECK (total >= 0),
    FOREIGN KEY (cliente_id) REFERENCES clientes(id)
);

-- 6. Detalle de pedido (líneas individuales)
CREATE TABLE detalle_pedido (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pedido_id INTEGER NOT NULL,
    producto_id INTEGER NOT NULL,
    cantidad INTEGER NOT NULL CHECK (cantidad > 0),
    precio_unitario REAL NOT NULL CHECK (precio_unitario >= 0),
    FOREIGN KEY (pedido_id) REFERENCES pedidos(id) ON DELETE CASCADE,
    FOREIGN KEY (producto_id) REFERENCES productos(id)
);

-- Índices para rendimiento en joins frecuentes
CREATE INDEX idx_productos_categoria ON productos(categoria_id);
CREATE INDEX idx_productos_proveedor ON productos(proveedor_id);
CREATE INDEX idx_pedidos_cliente ON pedidos(cliente_id);
CREATE INDEX idx_detalle_pedido ON detalle_pedido(pedido_id);
CREATE INDEX idx_detalle_producto ON detalle_pedido(producto_id);
```

Nota la diferencia con el primer intento:

```bash
$ sqlite3 techstore.db < schema.sql
$ sqlite3 techstore.db
sqlite> .tables
categorias      clientes        detalle_pedido  pedidos         productos       proveedores
```

7 tablas especializadas > 1 tabla monstruo. Este diseño cumple **1NF + 2NF + 3NF** y está listo para [[Advanced Querying|consultas avanzadas]] sin anomalías.

---

## 6. Denormalización controlada

A veces la normalización estricta **duele en performance**. Los motivos para **desnormalizar**:

- **Reporting pesado**: calcular `total` en cada SELECT con JOIN + SUM es lento. Lo pre-calculamos en `pedidos.total` (ya lo hicimos arriba).
- **READ-heavy**: si lees 1000 veces más de lo que escribes, duplicar datos acelera lecturas.
- **Cache de datos frecuentes**: guardar `cliente_nombre` en `pedidos` evita un JOIN, pero viola 3NF.

Ejemplo controlado:

```sql
-- Desnormalización aceptable: guardar precio_unitario en detalle_pedido
-- aunque ya existe en productos. El precio puede cambiar, pero la
-- línea de pedido debe reflejar el precio en el momento de la compra.
```

**Regla de oro**: normaliza hasta 3NF primero. Solo desnormaliza cuando **midas** un cuello de botella y la ganancia justifique la redundancia. [[Query Optimization & Indexing|La optimización sin medición es conjetura]].

---

## 7. Common Mistakes

| Error | Explicación |
|-------|-------------|
| Many-to-many sin tabla puente | `productos` y `proveedores`: un producto puede tener varios proveedores y viceversa. Sin `productos_proveedores` (tabla puente) no puedes modelarlo. |
| Missing indexes on FKs | Cada `FOREIGN KEY` es candidata a índice. Sin índice, los JOINs hacen escaneos completos (full table scan). |
| ON DELETE CASCADE sin entenderlo | Borrar un cliente elimina sus pedidos y detalles. Útil, pero peligroso si no lo controlas. Prefiere `ON DELETE RESTRICT` o borrado lógico (flag `activo`). |
| Usar `AUTOINCREMENT` sin necesidad | SQLite ya auto-incrementa con `INTEGER PRIMARY KEY`. `AUTOINCREMENT` evita reusar IDs pero gasta recursos. |
| No planificar tipos de dato | Usar `TEXT` para fechas impide funciones de fecha nativas. Usa `DATE` o `DATETIME` cuando el motor lo soporte. |
| Ignorar CHECK constraints | `CHECK (precio > 0)` evita datos inválidos a nivel BD. No confíes solo en validación del frontend. |

---

## Resumen

1. Una tabla única con todo genera redundancia, anomalías de inserción/actualización/borrado y datos imposibles de mantener.
2. **1NF** exige valores atómicos, sin columnas repetitivas, y PK única.
3. **2NF** elimina dependencias parciales: cada atributo debe depender de la PK completa.
4. **3NF** elimina dependencias transitivas: cada tabla modela una sola entidad.
5. El diseño final usa 7 tablas normalizadas con FKs, CHECKs, e índices — listo para producción.
6. La desnormalización es una herramienta, no un accidente: primero normaliza, después optimiza midiendo.
7. Los errores más comunes (tablas puente faltantes, índices olvidados, cascadas mal entendidas) se evitan con disciplina de diseño.

---

## Check Your Understanding

1. ¿Qué forma normal viola una tabla donde `cliente_direccion` depende de `cliente_id` dentro de una tabla `pedidos`?
   <!-- 3NF — dependencia transitiva: cliente_direccion no depende de la PK de pedidos -->

2. ¿Por qué una relación many-to-many necesita una tabla puente?
   <!-- Porque sin ella tendrías que repetir datos o usar columnas múltiples; la tabla puente almacena cada combinación como una fila independiente -->

3. ¿Cuándo está justificado violar 3NF?
   <!-- Cuando un perfil de rendimiento medido muestra que el JOIN es un cuello de botella y la redundancia controlada acelera lecturas críticas -->

4. ¿Qué diferencia hay entre `ON DELETE CASCADE` y `ON DELETE RESTRICT`?
   <!-- CASCADE borra en cascada las filas relacionadas; RESTRICT impide borrar si existen referencias -->

5. En la tabla `detalle_pedido`, ¿por qué guardamos `precio_unitario` si ya existe en `productos`?
   <!-- Porque el precio del producto puede cambiar (inflación, ofertas), pero el precio histórico del pedido debe congelarse al momento de la compra -->

---

## Where to Go Next

- [[Relational Model & SQL Fundamentals]] — si necesitas repasar CREATE TABLE, JOINs y GROUP BY
- [[Advanced Querying]] — subqueries, CTEs, window functions sobre esquemas normalizados
- [[Query Optimization & Indexing]] — cómo los índices aceleran los JOINs entre tablas normalizadas
- [[Data Engineering]] — pipelines ETL, modelado dimensional (star schema), data warehouses
- [[NoSQL Overview]] — cuándo NO usar SQL y qué alternativas existen
