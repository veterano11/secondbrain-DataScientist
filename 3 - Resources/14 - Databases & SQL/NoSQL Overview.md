---
tags:
  - nosql
  - databases
  - mongodb
  - redis
  - neo4j
  - cassandra
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu equipo arranca tres features al mismo tiempo: (1) un catálogo de productos donde cada categoría tiene atributos distintos (un laptop tiene "pulgadas_pantalla", una camiseta tiene "talle"), (2) un sistema de sesiones de usuario que necesita responder en milisegundos, y (3) un feed social tipo "a usuarios similares les gustó esto". El CTO dice "usemos SQL para todo, es lo que conocemos". Pero sabes que [[Relational Model & SQL Fundamentals|SQL relacional]] exige esquemas rígidos, JOINs costosos para grafos, y no está diseñado para caching ultrarápido. Necesitas conocer las alternativas NoSQL, cuándo usarlas y cuándo NO.

**Requisitos**: comprender los límites de SQL relacional ([[Relational Model & SQL Fundamentals]] y [[Database Design & Normalization]] ayudan).

---

## 1. ¿Por qué NoSQL?

SQL relacional es excelente para datos estructurados con relaciones claras y ACID. Pero tiene limitaciones:

| Escenario | Problema con SQL |
|-----------|-----------------|
| Schema flexible | Cada fila necesita los mismos campos. Agregar un atributo nuevo = `ALTER TABLE` + migración. |
| Alta escalabilidad horizontal | SQL escala verticalmente (más RAM/CPU). Sharding es posible pero complejo. |
| Caching de lecturas rápidas | SQL guarda en disco, no en RAM. Redis es 1000× más rápido para key-value. |
| Relaciones tipo grafo | Amigos-de-amigos requiere recursividad o múltiples JOINs. Neo4j lo hace en un salto. |
| Time-series (logs, métricas) | INSERT intensivo con muchos writes — SQL row-based no está optimizado para columnar. |

NoSQL no es "mejor" que SQL. Son herramientas diferentes. El concepto de **polyglot persistence** (usar la BD correcta para cada problema) es clave en [[Data Engineering|arquitecturas de datos modernas]].

```bash
# Verifica que tienes acceso a los clientes (no es necesario instalarlos)
$ which mongosh redis-cli cqlsh 2>/dev/null || echo "No todos los clientes están instalados"
```

---

## 2. Document stores (MongoDB)

Los **document stores** guardan datos en documentos JSON/BSON, no en filas y columnas. Cada documento puede tener su propia estructura (schemaless).

MongoDB es el más popular. Ejemplo de documento de producto con atributos flexibles:

```json
{
  "_id": ObjectId("abc123"),
  "nombre": "Laptop Gamer",
  "precio": 25000,
  "categoria": "Electrónica",
  "atributos": {
    "pulgadas_pantalla": 15.6,
    "ram_gb": 32,
    "almacenamiento": "1TB SSD"
  },
  "tags": ["gamer", "laptop", "ofertas"],
  "stock": 15
}
```

Otro producto en la misma colección tiene atributos completamente distintos:

```json
{
  "_id": ObjectId("def456"),
  "nombre": "Camiseta Algodón",
  "precio": 350,
  "categoria": "Ropa",
  "atributos": {
    "talle": "L",
    "color": "Negro",
    "material": "Algodón orgánico"
  },
  "tags": ["ropa", "básicos"],
  "stock": 200
}
```

**Query típica** — buscar productos con precio > 1000:

```javascript
db.productos.find({ precio: { $gt: 1000 } })
```

**Salida esperada:**

```json
[
  { "_id": ObjectId("abc123"), "nombre": "Laptop Gamer", "precio": 25000, ... },
  { "_id": ObjectId("ghi789"), "nombre": "Monitor 27\"", "precio": 8500, ... }
]
```

**Cuándo usar document stores:**
- Catálogos de productos con atributos variables (e-commerce, CMS)
- Logs de aplicación (eventos con estructura flexible)
- Datos que naturalmente son anidados (JSON anidado = documento)

**Cuándo NO:** cuando necesitas JOINs frecuentes entre colecciones, actualizaciones atómicas en múltiples documentos, o integridad referencial estricta.

MongoDB permite **embedding** (anidar arrays/subdocumentos) vs **referencing** (usar IDs como FK). La decisión depende de tus [[Advanced Querying|patrones de consulta]].

---

## 3. Key-Value stores (Redis)

Los **key-value stores** son la BD más simple posible: un mapa hash gigante en memoria. Redis es el rey indiscutido.

**Rendimiento**: operaciones en microsegundos (< 1ms). Ideal para datos accedidos por clave única.

```bash
$ redis-cli
127.0.0.1:6379> SET usuario:42:sesion "token=xyz&expires=2026-06-29T00:00:00Z"
OK
127.0.0.1:6379> GET usuario:42:sesion
"token=xyz&expires=2026-06-29T00:00:00Z"
127.0.0.1:6379> TTL usuario:42:sesion
(integer) 3599
127.0.0.1:6379> EXPIRE usuario:42:sesion 3600
(integer) 1
```

**Casos de uso principales:**

- **Caching**: resultados de queries SQL costosas. Guardas el resultado de un `SELECT ... JOIN ... GROUP BY` en Redis con TTL de 5 minutos.
- **Session store**: sesiones de usuario en web apps — `GET session:{id}` en < 1ms vs consultar SQL.
- **Rate limiting**: contadores atómicos con `INCR` + `EXPIRE`.
- **Leaderboards**: sorted sets para rankings en tiempo real.

```bash
127.0.0.1:6379> ZADD leaderboard 1500 "ana" 1200 "luis" 900 "carla"
(integer) 3
127.0.0.1:6379> ZREVRANGE leaderboard 0 2 WITHSCORES
1) "ana"
2) "1500"
3) "luis"
4) "1200"
5) "carla"
6) "900"
```

**Cuándo NO:** datos mayores a RAM disponible, queries complejas (más allá de key lookup), necesidad de persistencia ACID fuerte (Redis admite persistencia, pero no es su fuerte).

Redis tiene tipos de datos avanzados: **lists**, **sets**, **sorted sets**, **hashes**, **bitmaps**, **streams**. Cada uno resuelve un problema específico. [[CLI & Productivity|La terminal]] con `redis-cli` es tu mejor aliada para debuggear.

---

## 4. Column-family stores (Cassandra, BigTable)

Los **column-family stores** organizan datos por columnas en lugar de filas. Están diseñados para **escritura intensiva** y **escalabilidad horizontal** masiva.

Cassandra es el estándar. Ejemplo de schema para time-series (métricas de servidor):

```sql
CREATE TABLE metrics (
    server_id UUID,
    timestamp TIMESTAMP,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_io FLOAT,
    PRIMARY KEY (server_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);
```

**Insertar datos — millones de writes por segundo:**

```sql
INSERT INTO metrics (server_id, timestamp, cpu_usage, memory_usage, disk_io)
VALUES (uuid(), toTimestamp(now()), 78.5, 65.2, 120.0);
```

**Query típica** — última hora de métricas de un servidor:

```sql
SELECT * FROM metrics
WHERE server_id = ? AND timestamp >= '2026-06-28T10:00:00Z'
ORDER BY timestamp DESC;
```

**Salida esperada:**

```
server_id                           | timestamp                       | cpu_usage | memory_usage | disk_io
------------------------------------+--------------------------------+-----------+--------------+--------
550e8400-e29b-41d4-a716-446655440000| 2026-06-28 10:45:00.000+0000  | 82.1      | 70.3         | 145.2
550e8400-e29b-41d4-a716-446655440000| 2026-06-28 10:40:00.000+0000  | 78.5      | 65.2         | 120.0
550e8400-e29b-41d4-a716-446655440000| 2026-06-28 10:35:00.000+0000  | 75.0      | 62.8         | 110.5
```

**Cuándo usar column-family:**
- Time-series: logs, métricas IoT, eventos de servidor
- Escritura masiva (millones de inserts por segundo)
- Datos que siempre se leen por "rango" (timestamp, ID de sensor)
- Escalabilidad horizontal automática (agregar nodos sin downtime)

**Cuándo NO:** joins, agregaciones complejas, transacciones ACID entre filas, esquemas que cambian frecuentemente.

El diseño en Cassandra se guía por **query-first**: piensas las queries primero y modelas las tablas para responderlas. Esto es opuesto a [[Database Design & Normalization|normalización SQL]], donde primero diseñas el esquema y después escribes queries.

---

## 5. Graph databases (Neo4j)

Las **graph databases** modelan datos como nodos y relaciones. Brillan cuando la **relación entre datos es tan importante como los datos mismos**.

Neo4j usa **Cypher**, un lenguaje de consulta declarativo para grafos.

Ejemplo — red social con relaciones de amistad y gustos:

```cypher
CREATE (ana:Usuario {nombre: "Ana", edad: 28})
CREATE (luis:Usuario {nombre: "Luis", edad: 32})
CREATE (carla:Usuario {nombre: "Carla", edad: 25})
CREATE (laptop:Producto {nombre: "Laptop Gamer", precio: 25000})
CREATE (mouse:Producto {nombre: "Mouse Pro", precio: 1200})

CREATE (ana)-[:AMIGA_DE]->(luis)
CREATE (luis)-[:AMIGA_DE]->(carla)
CREATE (ana)-[:LE_GUSTA]->(laptop)
CREATE (luis)-[:LE_GUSTA]->(laptop)
CREATE (carla)-[:LE_GUSTA]->(mouse)
CREATE (ana)-[:LE_GUSTA]->(mouse)
```

**Query: recomendar productos que le gustaron a amigos de mis amigos:**

```cypher
MATCH (yo:Usuario {nombre: "Ana"})-[:AMIGA_DE*2]->(amigos)
MATCH (amigos)-[:LE_GUSTA]->(producto)
WHERE NOT (yo)-[:LE_GUSTA]->(producto)
RETURN DISTINCT producto.nombre, COUNT(amigos) as votos
ORDER BY votos DESC;
```

**Salida esperada:**

```
producto.nombre|votos
Mouse Pro      |2
```

**Cuándo usar graph databases:**
- Redes sociales (amigos-de-amigos, recomendaciones)
- Detección de fraude (transacciones sospechosas conectadas)
- Motores de recomendación (a usuarios similares les gustó X)
- Gestión de dependencias (paquetes, microservicios, pipelines)
- Data lineage (¿qué pipeline generó este dataset?)

**Cuándo NO:** datos puramente tabulares sin relaciones profundas, alta volumetría de escritura simple, casos donde SQL con JOINs alcanza.

---

## 6. Polyglot persistence

Las empresas no eligen "una BD para gobernarlos a todos". Usan varias según el caso. Ejemplo de arquitectura real:

```
┌─────────────────────────────────────────────────────┐
│                   API Gateway                       │
├─────────┬──────────┬──────────┬──────────┬──────────┤
│  MySQL  │ MongoDB  │  Redis   │Cassandra │ Neo4j    │
│ (pedidos│(catálogo │ (sesión, │ (logs,   │ (recomen-│
│  + ACID)│ flexible)│  cache)  │ métricas)│ daciones)│
└─────────┴──────────┴──────────┴──────────┴──────────┘
```

- **MySQL/PostgreSQL**: transacciones de pedidos (necesitan ACID).
- **MongoDB**: catálogo de productos con atributos variables.
- **Redis**: sesiones de usuario en < 1ms + cache de consultas SQL pesadas.
- **Cassandra**: logs de acceso y métricas de servidor (escritura intensiva).
- **Neo4j**: motor de recomendaciones y detección de fraude en red social.

Cada BD hace lo que mejor sabe hacer. La aplicación se comunica con todas a través de una capa de abstracción (repositorios/DAOs). [[Python for Data Science|Python]] con SQLAlchemy + pymongo + redis-py + cassandra-driver + neo4j-driver permite interactuar con todas desde el mismo código.

Este enfoque es estándar en [[Data Engineering|ingeniería de datos moderna]].

---

## 7. Common Mistakes

| Error | Explicación |
|-------|-------------|
| Usar NoSQL "porque es trendy" | Elegir MongoDB para un sistema contable (que necesita ACID) es desastre. NoSQL no es "SQL pero mejor". |
| Ignorar ACID cuando lo necesitas | Pagos, pedidos, inventario — si pierdes consistencia, pierdes dinero. SQL transaccional gana aquí. |
| Desnormalizar sin conocer los query patterns | En MongoDB, si siempre consultas por `cliente_id` pero modelas embedding por `producto_id`, cada query escanea toda la colección. Diseña según las queries, no al revés. |
| Key-value para todo | Redis no es una base de datos relacional. Meter datos relacionales en Redis (ej: todas las órdenes de un cliente como keys separadas sin relación) es receta para código espagueti. |
| Schema-less = sin schema | MongoDB no valida esquemas automáticamente, pero tu código sí espera ciertos campos. Documenta y valida en la app o usa MongoDB Schema Validation. |
| Ignorar el modelo de consistencia | Cassandra es eventualmente consistente por defecto. Si lees justo después de escribir, puedes leer datos viejos. Conoce tu modelo de consistencia antes de diseñar. |

---

## Resumen

1. SQL relacional tiene límites en schema flexible, escalabilidad horizontal, caching, grafos y time-series — NoSQL cubre esos casos.
2. **Document stores (MongoDB)** : schemaless, embedding vs referencing, ideal para catálogos con atributos variables.
3. **Key-Value stores (Redis)** : en memoria, microsegundos, sesiones, caching, rate limiting, leaderboards.
4. **Column-family stores (Cassandra)** : escritura masiva, time-series, escalabilidad horizontal automática.
5. **Graph databases (Neo4j)** : relaciones complejas, recomendaciones, fraude, data lineage.
6. **Polyglot persistence** : la mayoría de sistemas en producción usan múltiples BDs, cada una para lo que mejor sabe hacer.
7. Los errores más comunes son usar NoSQL por moda, ignorar ACID cuando se necesita, y diseñar sin considerar los patrones de consulta reales.

---

## Check Your Understanding

1. ¿Cuándo es preferible MongoDB a PostgreSQL?
   <!-- Cuando los documentos tienen estructura variable (catálogo de productos, CMS) o necesitas embedding natural de datos anidados -->

2. ¿Qué tipo de NoSQL usarías para un leaderboard en tiempo real?
   <!-- Key-Value (Redis) con sorted sets (ZADD, ZREVRANGE) — responde en microsegundos con ordenamiento nativo -->

3. ¿Qué significa "polyglot persistence"?
   <!-- Usar múltiples bases de datos en un mismo sistema, cada una optimizada para un tipo de carga (SQL para transacciones, Redis para cache, Neo4j para grafos, etc.) -->

4. ¿Por qué Cassandra no es buena opción para un sistema bancario?
   <!-- Cassandra es eventualmente consistente por defecto. Los sistemas bancarios necesitan ACID fuerte (consistencia inmediata en lecturas después de escrituras) -->

5. ¿Qué lenguaje usa Neo4j para consultar grafos?
   <!-- Cypher, un lenguaje declarativo con sintaxis ASCII-art para patrones de nodos y relaciones -->

---

## Where to Go Next

- [[Relational Model & SQL Fundamentals]] — compara NoSQL con la base del modelo relacional
- [[Database Design & Normalization]] — entiende qué sacrificas al desnormalizar en NoSQL
- [[Data Engineering]] — arquitecturas de datos, pipelines, y polyglot persistence en producción
- [[AWS CodeBuild with Terraform]] — despliega infraestructura multi-BD en la nube
- [[Python for Data Science]] — conecta Python a MongoDB, Redis, Cassandra y Neo4j con drivers nativos
