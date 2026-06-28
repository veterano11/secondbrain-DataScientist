---
tags:
  - data-engineering
  - etl
  - pipelines
  - python
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tienes archivos CSV de ventas que llegan diariamente a un bucket S3. Cada archivo tiene un formato ligeramente distinto: a veces el separador es `;` en lugar de `,`, otras veces falta la columna `total`, y los tipos de datos cambian (`string` vs `int`). Necesitas extraerlos, transformarlos (limpiar, tipificar, agregar) y cargarlos en una base de datos. Esto es ETL.

**Requisitos**: Python 3.9+, pandas, sqlite3.

```bash
$ python --version
$ pip install pandas
```

---

## 1. ¿Qué es ETL?

ETL significa **Extract, Transform, Load** — el proceso backbone de la ingeniería de datos:

1. **Extract**: leer datos de fuentes (CSV, API, base de datos, S3).
2. **Transform**: limpiar, validar, enriquecer y agregar los datos.
3. **Load**: escribir los datos transformados en un destino (DB, data warehouse, lake).

Una variante común es **ELT** (Extract, Load, Transform): cargas los datos crudos primero y transformas después dentro del destino. Es típica en [[Data Warehousing & Lakehouse|data warehouses modernos]] como Snowflake o BigQuery.

---

## 2. Extract

El extracto debe ser **confiable** y **reproducible**. Cada fuente tiene su propio método.

### Leer CSV con pandas

```python
import pandas as pd

df = pd.read_csv("s3://bucket/ventas/2026-06-28.csv")
# Si el separador es distinto:
df = pd.read_csv("s3://bucket/ventas/2026-06-28.csv", sep=";")
```

### Leer desde una API

```python
import requests

resp = requests.get("https://api.tienda.com/ventas?fecha=2026-06-28")
resp.raise_for_status()  # lanza error si status != 200
data = resp.json()
df = pd.DataFrame(data)
```

### Manejo de errores y reintentos

```python
import time
from requests.exceptions import RequestException

def extract_with_retry(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))  # backoff exponencial
```

**Salida esperada**: el extract devuelve un DataFrame o lista de dicts. Siempre con manejo de errores y logging. [[CLI & Productivity|Automatizar con scripts de shell]] ayuda a orquestar extractos programados.

---

## 3. Transform

La transformación es donde ocurre la mayor parte del trabajo y donde se gasta la mayor parte del tiempo.

### Limpieza básica

```python
# Eliminar filas totalmente vacías
df = df.dropna(how="all")

# Llenar nulos en columnas numéricas con 0
df["total"] = df["total"].fillna(0.0)

# Convertir tipos
df["fecha"] = pd.to_datetime(df["fecha"])
df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
```

### Validación

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "cliente_id": pa.Column(int, pa.Check.ge(0)),
    "total": pa.Column(float, pa.Check.ge(0)),
    "fecha": pa.Column(pd.Timestamp),
})

schema.validate(df)  # lanza error si algún registro no cumple
```

### Enriquecimiento y agregación

```python
# Agregar categoría de producto desde lookup table
df = df.merge(categorias, on="producto_id", how="left")

# Agregar ventas por día
resumen_diario = df.groupby("fecha").agg(
    ventas_totales=("total", "sum"),
    num_pedidos=("pedido_id", "nunique"),
).reset_index()
```

**Salida esperada**: un DataFrame limpio con tipos correctos. La calidad aquí determina la [[Data Quality & Testing|confiabilidad del pipeline completo]].

---

## 4. Load

Cargar los datos transformados en el destino final.

### Insertar en SQLite

```python
import sqlite3

conn = sqlite3.connect("ventas.db")
df.to_sql("ventas_diarias", conn, if_exists="append", index=False)
```

### UPSERT (INSERT OR REPLACE)

```python
# SQLite: INSERT OR REPLACE
df.to_sql("ventas_diarias", conn, if_exists="append", index=False)
conn.execute("""
    INSERT OR REPLACE INTO ventas_diarias
    SELECT * FROM ventas_diarias_staging
""")
```

### Batch vs row-by-row

```python
# ✅ Batch insert (rápido)
df.to_sql("ventas", conn, if_exists="append", index=False, chunksize=5000)

# ❌ Row-by-row (lento) — evítalo
for _, row in df.iterrows():
    conn.execute("INSERT INTO ventas VALUES (?, ?, ?)", tuple(row))
```

Usar `to_sql` con `chunksize` es órdenes de magnitud más rápido que insertar fila por fila. [[Relational Model & SQL Fundamentals|Conocer SQL]] ayuda a optimizar el load.

---

## 5. Pipeline completo con Python

Un script ETL end-to-end con logging:

```python
import logging
import pandas as pd
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def extract(archivo: Path) -> pd.DataFrame:
    logging.info(f"Extrayendo {archivo}")
    return pd.read_csv(archivo)

def transform(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Transformando datos")
    df = df.dropna(how="all")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
    return df

def load(df: pd.DataFrame, db_path: str, table: str):
    logging.info(f"Cargando {len(df)} filas en {table}")
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=5000)
    conn.close()

if __name__ == "__main__":
    data = extract(Path("ventas_2026-06-28.csv"))
    data = transform(data)
    load(data, "ventas.db", "ventas_diarias")
    logging.info("Pipeline completado")
```

**Salida esperada**:
```
2026-06-28 10:00:01 - Extrayendo ventas_2026-06-28.csv
2026-06-28 10:00:02 - Transformando datos
2026-06-28 10:00:03 - Cargando 1450 filas en ventas_diarias
2026-06-28 10:00:04 - Pipeline completado
```

Este patrón se escala a [[Workflow Orchestration (Airflow)|DAGs de Airflow]] y [[ML Pipelines|pipelines de ML]] sin cambiar la lógica central.

---

## 6. Batch vs Streaming

| Característica | Batch | Streaming |
|---|---|---|
| Frecuencia | Cada hora / día / semana | Continua (milisegundos) |
| Latencia | Minutos a horas | Segundos |
| Volumen | Alto por lote | Bajo por evento |
| Complejidad | Baja | Alta |
| Procesamiento | Acumulativo | Evento por evento |

**Cuándo usar batch**: reportes financieros diarios, entrenamiento de modelos, históricos.

**Cuándo usar streaming**: alertas en tiempo real, dashboards en vivo, detección de anomalías. Ver [[Streaming & Event-Driven (Kafka)|Kafka]] para el enfoque de streaming.

---

## 7. Common Mistakes

| Error | Explicación |
|---|---|
| No manejar datos duplicados | Usa `drop_duplicates()` en transform o `INSERT OR REPLACE` en load. |
| Cargar sin validar | Un schema roto en la base es difícil de reparar. Valida antes de cargar. |
| Sin logging | Cuando el pipeline falla a las 3 AM, el logging es tu única traza. |
| Transformaciones en el load | `to_sql` con `dtype` incorrecto — transforma antes en pandas. |
| Sin manejo de errores | Un archivo corrupto no debe tumbar todo el pipeline. Usa `try/except` por fuente. |

El [[Python for Data Science]] mindset de "validar primero, cargar después" ahorra horas de depuración.

---

## Resumen

1. ETL es el proceso backbone: Extraer → Transformar → Cargar.
2. El **Extract** debe manejar errores y reintentos de forma robusta.
3. La **Transformación** incluye limpieza, validación, enriquecimiento y agregación.
4. El **Load** debe usar batch inserts y soportar upserts.
5. Un pipeline completo en Python une extract/transform/load con logging.
6. Batch es para procesamiento periódico; Streaming para tiempo real.
7. Los errores comunes tienen solución — valida, registra, no dupliques.

---

## Check Your Understanding

1. ¿Qué diferencia hay entre ETL y ELT?
   <!-- ETL transforma antes de cargar; ELT carga crudo y transforma dentro del destino. -->

2. ¿Por qué `df.iterrows()` es un antipatrón en el Load?
   <!-- Porque inserta fila por fila, que es órdenes de magnitud más lento que batch insert con to_sql y chunksize. -->

3. ¿Qué hace `errors="coerce"` en `pd.to_numeric`?
   <!-- Convierte valores inválidos a NaN en lugar de lanzar error. -->

4. Nombra dos fuentes comunes de extract y cómo manejarías errores en cada una.
   <!-- API: try/except con retry + backoff. Archivo: try/except FileNotFoundError, validar que el archivo existe y tiene el formato esperado. -->

5. ¿Cuándo elegirías streaming en lugar de batch?
   <!-- Cuando necesitas latencia de segundos: alertas en tiempo real, dashboards en vivo, detección de anomalías. -->

---

## Where to Go Next

- [[Workflow Orchestration (Airflow)]] — orquesta pipelines multi-paso con reintentos
- [[Streaming & Event-Driven (Kafka)]] — baja la latencia a tiempo real
- [[Data Quality & Testing]] — asegura que tus pipelines producen datos correctos
- [[Relational Model & SQL Fundamentals]] — SQL para el load y la validación
- [[Python for Data Science]] — pandas y numpy para transformaciones eficientes
