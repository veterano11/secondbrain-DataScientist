---
tags:
  - data-engineering
  - airflow
  - orchestration
  - pipelines
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu pipeline ETL tiene 8 pasos: extraer de 3 fuentes distintas, validar cada una, transformar, enriquecer con datos maestros, cargar en staging y hacer upsert final en producción. Cuando falla el paso 3, tienes que reiniciar todo desde cero. Los pasos exitosos se repiten innecesariamente. Necesitas un orquestador: Apache Airflow.

**Requisitos**: Docker (para Airflow), Python 3.9+. Opcional: `apache-airflow` instalado.

```bash
$ docker --version
```

---

## 1. ¿Qué es orquestación?

Orquestar es **coordinar tareas** con dependencias, reintentos, scheduling y monitoreo. No es ETL — es el "director de orquesta" que decide qué tarea corre, cuándo, en qué orden, y qué hacer si falla.

Un orquestador ofrece:

- **Scheduling**: "corre este DAG todos los días a las 6 AM".
- **Dependencias**: "no corras el paso 4 hasta que termine el paso 3".
- **Reintentos**: "si falla, reintenta 2 veces con espera de 5 minutos".
- **Alertas**: "si falla después de reintentos, mándame un Slack/email".

Apache Airflow es el orquestador open-source más usado en Data Engineering. Se integra con [[Data Pipelines & ETL]] y [[ML Pipelines]] de forma natural.

---

## 2. Conceptos clave

| Concepto | Definición |
|---|---|
| **DAG** | Directed Acyclic Graph — el pipeline completo (grafo acíclico de tareas). |
| **Task** | Una unidad de trabajo dentro del DAG (una función Python, un comando bash, etc.). |
| **Operator** | Plantilla de tarea: `PythonOperator`, `BashOperator`, `PostgresOperator`. |
| **Schedule** | Intervalo cron: `@daily`, `@hourly`, o expresión cron. |
| **Sensor** | Task que espera a que ocurra algo (archivo en S3, registro en DB). |

Cada DAG se define en un archivo Python que Airflow lee y parsea. [[Python for Data Science]] es la base para escribir Operators efectivos.

---

## 3. Tu primer DAG

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

def _extract():
    print("Extrayendo datos...")

def _transform():
    print("Transformando datos...")

def _load():
    print("Cargando datos...")

with DAG(
    dag_id="etl_simple",
    start_date=datetime(2026, 6, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    extract = PythonOperator(task_id="extract", python_callable=_extract)
    transform = PythonOperator(task_id="transform", python_callable=_transform)
    load = PythonOperator(task_id="load", python_callable=_load)

    extract >> transform >> load
```

**Salida esperada**: En la UI de Airflow, ves el DAG `etl_simple` con 3 tareas en secuencia. Cada día a medianoche corre automáticamente.

El `>>` define dependencias. Se lee: "extract → transform → load". [[CLI & Productivity|Scripts de bash]] se integran vía `BashOperator`.

---

## 4. Dependencias

### Dependencias básicas

```python
task_a >> task_b           # task_b depende de task_a
task_a >> [task_b, task_c]  # task_b y task_c dependen de task_a
[task_a, task_b] >> task_c  # task_c depende de task_a y task_b
```

### Dependencias complejas

```python
extract_api = PythonOperator(...)
extract_db = PythonOperator(...)
validate = PythonOperator(...)

extract_api >> validate
extract_db >> validate
validate >> [transform_enrich, transform_aggregate]
transform_enrich >> load
transform_aggregate >> load
```

### Cross-DAG dependencies

```python
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_ml_dag = ExternalTaskSensor(
    task_id="wait_for_training",
    external_dag_id="ml_training",
    external_task_id="train_model",
)

# Este DAG espera a que el DAG ml_training termine su task train_model
extract >> transform >> wait_for_ml_dag >> load
```

Las dependencias entre DAGs son poderosas pero deben usarse con moderación — prefieres un solo DAG limpio. [[Data Quality & Testing|Tests de DAGs]] ayudan a mantener la estructura.

---

## 5. Operators útiles

### PostgresOperator

```python
from airflow.providers.postgres.operators.postgres import PostgresOperator

create_table = PostgresOperator(
    task_id="create_table",
    postgres_conn_id="postgres_default",
    sql="CREATE TABLE IF NOT EXISTS ventas (...);",
)
```

### S3KeySensor

Espera a que un archivo aparezca en S3:

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_for_file = S3KeySensor(
    task_id="wait_for_csv",
    bucket_key="s3://bucket/ventas/{{ ds }}.csv",
    poke_interval=60,  # revisa cada 60 segundos
    timeout=3600,       # timeout de 1 hora
)
```

### PythonVirtualenvOperator

Corre tareas en un virtualenv aislado — útil para dependencias conflictivas:

```python
from airflow.operators.python import PythonVirtualenvOperator

def heavy_computation():
    import pandas as pd
    import numpy as np
    # ... código que necesita pandas

task = PythonVirtualenvOperator(
    task_id="compute",
    python_callable=heavy_computation,
    requirements=["pandas", "numpy"],
)
```

[[Docker Fundamentals]] es útil si prefieres contenedores completos en lugar de virtualenvs.

---

## 6. Monitoreo

### Airflow UI

La UI web de Airflow (puerto 8080 por defecto) muestra:

- Estado de cada DAG (running, failed, success, paused).
- Logs de cada task instance.
- Tree View y Graph View para visualizar dependencias.
- Gantt Chart para detectar cuellos de botella.

### Alertas

```python
from airflow.models import DAG

default_args = {
    "email": ["data-team@empresa.com"],
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="etl_ventas",
    default_args=default_args,
    schedule_interval="@daily",
) as dag:
    ...
```

Para Slack, usa el `SlackWebhookOperator` de los providers de Slack. [[Experiment Tracking]] comparte principios similares de logging y monitoreo.

---

## 7. Best practices

| Práctica | Explicación |
|---|---|
| **Tareas idempotentes** | Correr la misma tarea dos veces debe dar el mismo resultado. Usa upserts, no inserts simples. |
| **DAGs dinámicos** | Genera tareas con `for` loops: una task por fuente de datos, en lugar de un DAG estático. |
| **Variables y conexiones** | No hardcodees credenciales. Usa Airflow Variables y Connections. |
| **Tareas cortas** | Una task debe durar minutos, no horas. Divide tareas largas. |
| **Testing local** | Usa `dag.test()` para probar un DAG sin el scheduler. |

```python
# DAG dinámico: una task por archivo fuente
with DAG(dag_id="etl_dinamico", ...) as dag:
    for source in ["api", "db", "s3"]:
        extract = PythonOperator(task_id=f"extract_{source}", ...)
        validate = PythonOperator(task_id=f"validate_{source}", ...)
        extract >> validate
```

La idempotencia es el principio más importante — sin ella, los reintentos generan datos corruptos. [[Data Pipelines & ETL]] profundiza en este patrón.

---

## 8. Common Mistakes

| Error | Explicación |
|---|---|
| Tareas demasiado largas | Una task que dura 2 horas bloquea recursos y es difícil de depurar. Dividir. |
| No setear `retries` | Si no configuras reintentos, Airflow no reintenta automáticamente. |
| DAGs no idempotentes | Si el DAG corre dos veces el mismo día, genera datos duplicados. |
| `start_date` incorrecto | DAGs atrapados en backfill infinito — usa `catchup=False`. |
| Sin logging en las tasks | Las tasks fallan silenciosamente sin mensajes útiles. Usa `self.log.info()`. |

---

## Resumen

1. La **orquestación** coordina scheduling, dependencias, reintentos y alertas.
2. Airflow modela pipelines como **DAGs** de **tasks** usando **operators**.
3. Las **dependencias** se definen con `>>` (task_a >> task_b).
4. Operators comunes: `PythonOperator`, `BashOperator`, `PostgresOperator`, `S3KeySensor`.
5. El monitoreo usa la UI de Airflow + alertas por email/Slack.
6. Las mejores prácticas incluyen idempotencia, DAGs dinámicos y tareas cortas.
7. Los errores comunes tienen solución — configura reintentos, evita tareas largas.

---

## Comprueba tu Conocimiento

1. ¿Qué es un DAG y por qué debe ser acíclico?
   <!-- Un Directed Acyclic Graph define el pipeline. Es acíclico para evitar loops infinitos. -->

2. ¿Qué operador usarías para correr código de pandas que requiere dependencias específicas?
   <!-- PythonVirtualenvOperator, que permite aislar dependencias en un virtualenv. -->

3. ¿Qué hace `catchup=False` en un DAG?
   <!-- Evita que Airflow ejecute tareas atrasadas para fechas anteriores a start_date. -->

4. ¿Por qué las tareas deben ser idempotentes?
   <!-- Para que reintentar una tarea no genere duplicados ni datos corruptos. -->

5. ¿Cómo monitoreas si un DAG falló?
   <!-- Revisando la UI de Airflow (DAG Runs / Task Instances), logs, o configurando alertas (email_on_failure, Slack). -->

---

## ¿Dónde ir Siguente?

- [[Data Pipelines & ETL]] — construye los pipelines que Airflow va a orquestar
- [[Python for Data Science]] — Python para escribir operators efectivos
- [[Docker Fundamentals]] — corre Airflow y dependencias en contenedores
- [[ML Pipelines]] — orquesta pipelines de ML con Airflow
- [[Data Quality & Testing]] — tests y validación dentro de los DAGs
