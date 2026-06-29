---
tags: [polars, dataframe, performance, big-data, python]
status: seedling
created: 2026-06-28
---

# Polars for Data Science

## 1. Escenario de aprendizaje

Tienes un archivo CSV de 10GB con ventas de un año. Lo cargas con pandas y recibes `MemoryError`. Tu laptop tiene 16GB de RAM. Polars puede procesar ese archivo sin problemas, usando **lazy evaluation** y **ejecución paralela** en múltiples núcleos. Al terminar esta nota, podrás migrar pipelines de pandas a Polars y procesar datasets que antes no podías.

## 2. Requisitos

```
$ pip install polars
$ python -c "import polars as pl; print(pl.__version__)"
1.2.0
```

Polars requiere Python 3.8+. No necesita Java, Spark ni infraestructura distribuida — corre en una sola máquina aprovechando todos los núcleos.

## 3. ¿Qué hace diferente a Polars?

Polars no es "pandas con otra sintaxis". Hay diferencias fundamentales:

| Característica | Pandas | Polars |
|---|---|---|
| Evaluación | **Eager** (todo se ejecuta inmediatamente) | **Lazy** por defecto (construye un plan, optimiza, ejecuta) |
| Ejecución | Single-thread (1 núcleo) | **Paralelo** (todos los núcleos) |
| Memoria | Copia datos en cada operación | **Zero-copy** entre operaciones |
| API | `.apply()` con Python loops (lento) | **Expresiones** compiladas a Rust |
| Motor | NumPy + Python | **Rust** (Apache Arrow Columnar Format) |
| Streaming | No (todo en RAM) | Sí (procesa en chunks, RAM constante) |

```python
# pandas: opera en memoria, eager
import pandas as pd
df = pd.read_csv("ventas_10gb.csv")  # MemoryError si no alcanza RAM

# polars: lazy, optimiza, paraleliza
import polars as pl
df = pl.scan_csv("ventas_10gb.csv")  # no carga nada aún
```

## 4. DataFrames en Polars

Un `pl.DataFrame` es similar a un `pd.DataFrame` pero inmutable por defecto (las operaciones devuelven un nuevo DataFrame):

```python
import polars as pl

df = pl.DataFrame({
    "cliente": ["Ana", "Bob", "Cat", "Ana", "Bob"],
    "monto": [100, 250, 180, 300, 50],
    "fecha": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-03"]
})

print(df)
```

Salida esperada:
```
shape: (5, 3)
┌────────┬───────┬────────────┐
│ cliente ┆ monto ┆ fecha      │
│ ---    ┆ ---   ┆ ---        │
│ str    ┆ i64   ┆ str        │
╞════════╪═══════╪════════════╡
│ Ana    ┆ 100   ┆ 2026-01-01 │
│ Bob    ┆ 250   ┆ 2026-01-01 │
│ Cat    ┆ 180   ┆ 2026-01-02 │
│ Ana    ┆ 300   ┆ 2026-01-03 │
│ Bob    ┆ 50    ┆ 2026-01-03 │
└────────┴───────┴────────────┘
```

La tabla se muestra con tipo de dato (`i64`, `str`) incluido — útil para depuración.

### 4.1 Tipos de datos

Polars usa el sistema de tipos de Apache Arrow:

| Tipo Polars | Equivalente pandas | Descripción |
|---|---|---|
| `pl.Int64`, `pl.Int32` | `int64`, `int32` | Enteros con signo |
| `pl.Float64` | `float64` | Punto flotante |
| `pl.Utf8` | `object` (string) | Cadenas de texto |
| `pl.Date`, `pl.Datetime` | `datetime64` | Fechas y timestamps |
| `pl.List(pl.Int64)` | `object` (list) | Listas anidadas en columnas |
| `pl.Struct({"a": pl.Int64, "b": pl.Utf8})` | - | Columnas compuestas (como dicts) |

Los tipos `List` y `Struct` permiten datos anidados sin necesidad de normalizar — una ventaja sobre pandas.

## 5. Lectura y escritura

Polars lee archivos tabulares con una API similar a pandas pero con optimizaciones clave:

### 5.1 Lectura eager (todo en RAM)

```python
# CSV
df = pl.read_csv("ventas.csv")

# Parquet (mucho más rápido que CSV)
df = pl.read_parquet("ventas.parquet")

# Excel, JSON, Avro...
df = pl.read_excel("datos.xlsx")
```

### 5.2 Lectura lazy (bajo demanda, recomendado para archivos +1GB)

```python
# scan_csv NO carga los datos — solo registra el archivo
q = pl.scan_csv("ventas_10gb.csv")

# Podemos encadenar transformaciones sin ejecutar
q = q.filter(pl.col("monto") > 100)
q = q.group_by("cliente").agg(pl.col("monto").sum())

# Solo aquí se ejecuta: paralelo, optimizado
df_resultado = q.collect()
```

`scan_csv` infiere el schema desde las primeras filas (`infer_schema_length=100` por defecto). Para archivos grandes sin cabecera, especifica el schema manualmente:

```python
q = pl.scan_csv("ventas_sin_header.csv", has_header=False,
                new_columns=["fecha", "cliente", "monto", "producto"],
                dtypes={"monto": pl.Float64})
```

### 5.3 Escritura

```python
df.write_parquet("resultados.parquet")
df.write_csv("resultados.csv")  # también funciona
```

**Recomendación**: usa Parquet para guardar resultados intermedios. Ocupa ~80% menos espacio que CSV y preserva tipos de datos.

## 6. Transformaciones

La API de transformaciones usa **expresiones** (`pl.col()`, `pl.sum()`, `pl.when()`) que son compuestas y optimizadas por el query planner:

### 6.1 Filter, select, with_columns

```python
# Filtrar
df_filtered = df.filter(pl.col("monto") > 100)

# Seleccionar columnas
df_sel = df.select(["cliente", "monto"])

# Crear/modificar columnas
df = df.with_columns(
    (pl.col("monto") * 1.19).alias("monto_con_iva"),
    pl.col("fecha").str.to_date("%Y-%m-%d").alias("fecha_dt")
)
```

### 6.2 Group by y aggregations

```python
# Total por cliente
por_cliente = df.group_by("cliente").agg([
    pl.col("monto").sum().alias("total_gastado"),
    pl.col("monto").mean().alias("promedio_compra"),
    pl.col("fecha").count().alias("num_compras"),
    pl.col("monto").max().alias("compra_maxima")
])

print(por_cliente)
```

Salida esperada:
```
shape: (3, 5)
┌─────────┬──────────────┬─────────────────┬─────────────┬──────────────┐
│ cliente ┆ total_gastado ┆ promedio_compra ┆ num_compras ┆ compra_maxima│
│ ---     ┆ ---           ┆ ---             ┆ ---          ┆ ---          │
│ str     ┆ i64           ┆ f64             ┆ u32          ┆ i64          │
╞═════════╪═══════════════╪═════════════════╪══════════════╪═══════════════╡
│ Ana     ┆ 400           ┆ 200.0           ┆ 2            ┆ 300          │
│ Bob     ┆ 300           ┆ 150.0           ┆ 2            ┆ 250          │
│ Cat     ┆ 180           ┆ 180.0           ┆ 1            ┆ 180          │
└─────────┴───────────────┴─────────────────┴──────────────┴───────────────┘
```

### 6.3 Joins

```python
clientes = pl.DataFrame({
    "cliente": ["Ana", "Bob", "Cat", "Dan"],
    "ciudad": ["Santiago", "Valparaíso", "Santiago", "Concepción"]
})

resultado = df.join(clientes, on="cliente", how="left")
```

Polars ofrece `join` con métodos `inner`, `left`, `outer`, `cross`, `semi` y `anti`. Los joins están altamente optimizados y corren en paralelo.

### 6.4 Window functions

```python
df = df.with_columns(
    pl.col("monto").sum().over("cliente").alias("total_cliente"),
    pl.col("monto").rank("dense").over("cliente").alias("ranking_por_cliente")
)
```

`over()` es el equivalente a `GROUP BY` combinado con window function — evita joins.

## 7. API Lazy — el superpoder de Polars

La API lazy es donde Polars realmente brilla. Construyes un **query plan** y Polars lo optimiza antes de ejecutar:

```python
q = (pl.scan_csv("ventas_10gb.csv")
     .filter(pl.col("monto") > 0)
     .with_columns(pl.col("fecha").str.to_date("%Y-%m-%d"))
     .group_by(["cliente", pl.col("fecha").dt.month().alias("mes")])
     .agg(pl.col("monto").sum())
     .sort("mes")
    )

# Ver el plan optimizado
print(q.explain())
```

Salida esperada (plan simplificado):
```
FILTER [(col("monto")) > (0)] FROM
  CSV SCAN ventas_10gb.csv
WITH_COLUMNS:
    col("fecha").str.to_date()
  AGGREGATE
    [col("monto").sum()] BY [cliente, month]
  SORT BY [mes]
```

Luego ejecutas:

```python
df_resultado = q.collect()  # ejecución paralela
```

### 7.1 Streaming para datasets que no caben en RAM

Para archivos que ni siquiera caben después de optimización:

```python
q = pl.scan_csv("ventas_100gb.csv")
    .filter(pl.col("monto") > 100)
    .group_by("cliente")
    .agg(pl.col("monto").sum())

# Streaming: procesa en lotes, RAM constante
df_resultado = q.collect(streaming=True)
```

Con `streaming=True`, Polars procesa el dataset en chunks. El uso de RAM se mantiene constante (~200MB) sin importar el tamaño del archivo.

## 8. Expresiones

Las expresiones son el bloque constructor de Polars. Se componen y ejecutan en Rust sin pasar por Python:

```python
df.with_columns(
    # Condicional
    pl.when(pl.col("monto") > 200)
      .then(pl.lit("alto"))
      .otherwise(pl.lit("bajo"))
      .alias("categoria"),

    # Strings
    pl.col("cliente").str.to_uppercase().alias("cliente_mayus"),
    pl.col("cliente").str.contains("^A").alias("empieza_con_A"),

    # Fechas
    pl.col("fecha").dt.year().alias("anio"),
    pl.col("fecha").dt.month().alias("mes"),

    # Múltiples agregaciones con map
    pl.col("monto").map_branches(
        pl.col("monto").sum().alias("suma"),
        pl.col("monto").mean().alias("media")
    ),

    # Correr funciones de Python (lento — usar solo cuando sea necesario)
    pl.col("cliente").map_elements(lambda s: s[::-1],
                                    return_dtype=pl.Utf8).alias("cliente_reves")
)
```

**Regla**: mientras más uses expresiones nativas (`.str.*`, `.dt.*`, `.arr.*`), más rápido corre. `.map_elements()` cae a Python y pierde la ventaja de Polars.

## 9. Polars vs pandas — benchmarks

Prueba con un dataset de 5M filas × 10 columnas:

| Operación | Pandas | Polars | Aceleración |
|---|---|---|---|
| `group_by().sum()` | 2.4s | 0.12s | **20x** |
| `join` (inner) | 1.8s | 0.09s | **20x** |
| `filter` (string) | 0.9s | 0.04s | **22x** |
| Carga CSV | 8.3s | 1.1s | **7.5x** |
| Carga Parquet | 1.2s | 0.3s | **4x** |

Polars es consistentemente más rápido gracias a:
- **Ejecución paralela** (usa todos los núcleos)
- **Código nativo en Rust** (sin GIL de Python)
- **Optimización de queries** (reordena operaciones, elimina columnas innecesarias)
- **Formato columnar Arrow** (mejor cache locality)

### ¿Cuándo NO usar Polars?

| Caso | Quedarse con pandas | Motivo |
|---|---|---|
| Datasets pequeños (<100MB) | pandas es suficiente | La diferencia es imperceptible |
| Código legacy con pandas API | pandas | Migrar tiene costo |
| Dependencia de `.apply()` con lógica compleja | pandas (o evaluar) | Polars no tiene apply nativo rico |
| Visualización exploratoria | pandas + matplotlib | Polars no tiene plotting integrado |
| Notebooks interactivos rápidos | pandas | Menos boilerplate |

## 10. Common Mistakes

- **No usar `scan_csv` para archivos grandes**: usar `read_csv` carga todo en RAM. Siempre prefiere `scan_csv` + `collect()` para datasets de GB.
- **Mezclar lazy y eager sin necesidad**: si usas `scan_csv` pero llamas `collect()` después de cada paso, pierdes la optimización. Construye todo el query plan y colecta una vez.
- **Usar `.map_elements()` en lugar de expresiones**: cada llamada a `map_elements` cae a Python. Prefiere `.str.*`, `.dt.*`, `.arr.*` nativas.
- **No especificar schema en archivos grandes sin cabecera**: Polars infiere leyendo las primeras filas. En archivos de 10GB sin cabecera, especifica `dtypes` o `infer_schema_length=0` y define el schema manualmente.
- **Comparar con pandas usando `.values`**: Polars no usa NumPy por defecto. Usa `.to_numpy()` si necesitas interfaz con NumPy.
- **Olvidar que Polars es inmutable**: `df[col] = valor` no funciona. Usa `df.with_columns(...)`.

## Resumen

1. Polars es un DataFrame engine en Rust que ejecuta en paralelo multiplataforma.
2. `pl.scan_csv()` + `collect()` reemplaza `pd.read_csv()` para archivos grandes.
3. Las expresiones (`pl.col()`) son el bloque constructor — prefierelas sobre funciones Python.
4. La API lazy optimiza el query plan antes de ejecutar.
5. `streaming=True` para datasets que no caben en RAM.
6. Polars es 10-20x más rápido que pandas en operaciones comunes.
7. No abandones pandas — usa Polars cuando necesites velocidad o trabajar con datasets +1GB.

## Check Your Understanding

1. ¿Cuál es la diferencia entre `pl.read_csv()` y `pl.scan_csv()`? <!-- `read_csv` carga todo en RAM inmediatamente; `scan_csv` registra el archivo para lectura lazy, y los datos se cargan solo cuando llamas `collect()`. -->

2. ¿Qué ventaja tiene `streaming=True` en `collect()`? <!-- Procesa el dataset en chunks, manteniendo el uso de RAM constante (~200MB) independientemente del tamaño del archivo. -->

3. ¿Cuándo usarías `.map_elements()` y por qué deberías evitarlo? <!-- Cuando necesitas una función Python que no se puede expresar con expresiones nativas. Debes evitarlo porque cae a Python y pierde la ventaja de velocidad de Polars. -->

4. ¿Qué hace `q.explain()`? <!-- Muestra el query plan optimizado que Polars ejecutará, útil para depurar y entender qué operaciones se realizan. -->

5. ¿Por qué Polars es más rápido que pandas en group_by? <!-- Porque ejecuta en paralelo en todos los núcleos (Rust + Arrow), mientras pandas corre en un solo hilo (NumPy + Python). -->

## Where to Go Next

- [[Python for Data Science]] — pandas, numpy, el ecosistema base
- [[Data Pipelines & ETL]] — pipelines de datos con Polars
- [[Feature Engineering]] — transformaciones de features a gran escala
- [[Data Quality & Testing]] — validación de datos en pipelines
- [[Advanced Querying]] — window functions en SQL (conceptos similares a `over()`)
- [[Data Warehousing & Lakehouse]] — formatos columnar y Parquet
- [[Virtual Environments]] — gestionar dependencias sin conflictos
