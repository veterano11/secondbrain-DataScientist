---
tags:
  - data-engineering
  - streaming
  - kafka
  - event-driven
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tu negocio necesita saber **en tiempo real** cuando un cliente hace una compra grande (> $10,000), o cuando un sensor reporta una temperatura anómala. Batch (procesar cada hora) no es suficiente — para cuando cargas los datos ya es demasiado tarde. Necesitas streaming con Apache Kafka.

**Requisitos**: Docker (para levantar Kafka local), Python 3.9+, `confluent-kafka`.

```bash
$ docker --version
$ pip install confluent-kafka
```

---

## 1. Batch vs Streaming

### Comparación Visual

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BATCH vs STREAMING                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BATCH (procesamiento por lotes)                                   │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                 │
│  │ Lote│ │ Lote│ │ Lote│ │ Lote│  ← Cada hora/día/semana         │
│  │  1  │ │  2  │ │  3  │ │  4  │                                 │
│  └─────┘ └─────┘ └─────┘ └─────┘                                 │
│      │        │        │        │                                   │
│      ▼        ▼        ▼        ▼                                   │
│  ┌─────────────────────────────────────┐                          │
│  │         Procesamiento               │                          │
│  └─────────────────────────────────────┘                          │
│                                                                     │
│  Latencia: minutos a horas                                        │
│  Throughput: alto (procesa lotes grandes)                          │
│  Ejemplo: reportes diarios, ETL nocturno                          │
│                                                                     │
│  ─────────────────────────────────────────────────────────────     │
│                                                                     │
│  STREAMING (procesamiento en tiempo real)                         │
│  ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐                       │
│  │ E ││ E ││ E ││ E ││ E ││ E ││ E ││ E │  ← Cada evento         │
│  └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘                       │
│      │    │    │    │    │    │    │    │                           │
│      ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼                           │
│  ┌─────────────────────────────────────┐                          │
│  │         Procesamiento               │                          │
│  └─────────────────────────────────────┘                          │
│                                                                     │
│  Latencia: milisegundos a segundos                                │
│  Throughput: alto (procesa eventos individuales rápido)            │
│  Ejemplo: alertas, dashboards en vivo, fraudes                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Característica | Batch | Streaming |
|---|---|---|
| Latencia | Minutos a horas | Milisegundos a segundos |
| Procesamiento | Por lotes programados | Evento por evento, en tiempo real |
| Throughput | Alto (procesa lotes grandes) | Alto (procesa eventos individuales rápido) |
| Exactly-once | Sencillo (transacciones batch) | Más complejo (offsets + idempotencia) |
| Casos de uso | Reportes diarios, ETL | Alertas, dashboards en vivo, fraudes |

En streaming, **exactly-once semantics** significa que cada evento se procesa exactamente una vez, ni más ni menos. Es más difícil que en batch porque los eventos pueden fallar, reintentarse, o procesarse fuera de orden.

Kafka es la plataforma de streaming más adoptada. [[Data Pipelines & ETL|Los pipelines ETL tradicionales]] y Kafka a menudo coexisten — streaming para lo urgente, batch para lo analítico.

---

## 2. Kafka concepts

### Arquitectura Visual de Kafka

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA KAFKA                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PRODUCERS              KAFKA CLUSTER              CONSUMERS       │
│  ┌─────────┐       ┌─────────────────────┐       ┌─────────┐      │
│  │ App 1   │──────▶│  ┌─────────────┐   │──────▶│ App 1   │      │
│  └─────────┘       │  │   Topic     │   │       └─────────┘      │
│  ┌─────────┐       │  │  "compras"  │   │       ┌─────────┐      │
│  │ App 2   │──────▶│  │             │   │──────▶│ App 2   │      │
│  └─────────┘       │  │ P0: [e1,e2] │   │       └─────────┘      │
│  ┌─────────┐       │  │ P1: [e3,e4] │   │       ┌─────────┐      │
│  │ App 3   │──────▶│  │ P2: [e5,e6] │   │──────▶│ App 3   │      │
│  └─────────┘       │  └─────────────┘   │       └─────────┘      │
│                    │                     │                          │
│                    │  ┌─────────────┐   │                          │
│                    │  │  Brokers    │   │                          │
│                    │  │  (3 nodos)  │   │                          │
│                    │  └─────────────┘   │                          │
│                    └─────────────────────┘                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Concepto | Definición |
|---|---|
| **Topic** | Canal de datos — como una "categoría" de eventos (ej: `compras`, `sensores`). |
| **Partition** | Cada topic se divide en particiones para escalar horizontalmente. |
| **Producer** | Aplicación que **escribe** eventos a un topic. |
| **Consumer** | Aplicación que **lee** eventos de un topic. |
| **Consumer Group** | Grupo de consumers que dividen las particiones entre sí. |
| **Offset** | Número secuencial que identifica la posición de un evento en una partición. |

```
                   Topic "compras"
    ┌────────────────────────────────────────┐
    │ Partition 0: [ev1, ev2, ev3, ev4, ...] │
    │ Partition 1: [ev5, ev6, ev7, ...]      │
    │ Partition 2: [ev8, ev9, ...]           │
    └────────────────────────────────────────┘
         ▲                          ▲
         │                          │
    Producer                    Consumer Group "alertas"
                                ├── consumer-1 (Partition 0)
                                ├── consumer-2 (Partition 1, 2)
```

Cada consumer en un grupo lee de particiones exclusivas — nunca dos consumers en el mismo grupo leen la misma partición. [[Docker Fundamentals]] es la forma más común de levantar Kafka localmente.

---

## 3. Kafka CLI

### Crear un topic

```bash
$ kafka-topics --bootstrap-server localhost:9092 \
    --create --topic compras \
    --partitions 3 --replication-factor 1
```

### Producir mensajes

```bash
$ kafka-console-producer --bootstrap-server localhost:9092 \
    --topic compras
> {"cliente": "Ana", "total": 15000, "producto": "Laptop"}
> {"cliente": "Luis", "total": 350, "producto": "Mouse"}
```

### Consumir mensajes

```bash
$ kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic compras --from-beginning
```

**Salida esperada**:
```
{"cliente": "Ana", "total": 15000, "producto": "Laptop"}
{"cliente": "Luis", "total": 350, "producto": "Mouse"}
```

### Ver el estado del grupo

```bash
$ kafka-consumer-groups --bootstrap-server localhost:9092 \
    --group alertas --describe
```

Muestra el **consumer lag** — cuántos eventos tiene atrasados cada consumer. Monitorear esto es crítico. [[CLI & Productivity|Dominar la terminal]] acelera mucho la depuración de Kafka.

---

## 4. Producer/Consumer en Python

### Producer

```python
from confluent_kafka import Producer
import json

def delivery_report(err, msg):
    if err is not None:
        print(f"Error al enviar: {err}")
    else:
        print(f"Enviado a {msg.topic()} [{msg.partition()}]")

producer = Producer({"bootstrap.servers": "localhost:9092"})

event = {"cliente": "Ana", "total": 15000, "producto": "Laptop"}
producer.produce(
    topic="compras",
    value=json.dumps(event).encode("utf-8"),
    callback=delivery_report,
)
producer.flush()  # esperar a que se envíe
```

`acks` controla la durabilidad:
- `acks=0`: no espera confirmación (más rápido, menos seguro).
- `acks=1`: espera confirmación del líder (default).
- `acks=all`: espera confirmación de todas las réplicas (más seguro).

### Consumer

```python
from confluent_kafka import Consumer, KafkaError

consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "alertas",
    "auto.offset.reset": "earliest",
})

consumer.subscribe(["compras"])

while True:
    msg = consumer.poll(1.0)  # timeout de 1 segundo
    if msg is None:
        continue
    if msg.error():
        print(f"Error: {msg.error()}")
        continue

    event = json.loads(msg.value().decode("utf-8"))
    if event["total"] > 10000:
        print(f"🚨 Compra grande: {event}")
    # Commit manual del offset
    consumer.commit(msg)
```

El commit manual (`consumer.commit()`) es más seguro que el auto-commit — así evitas perder eventos si el consumer crashea. [[Python for Data Science|Patterns de Python]] como context managers y callbacks hacen el código más robusto.

### Consumer lag

```python
# Monitorear lag desde la línea de comandos
# kafka-consumer-groups --bootstrap-server localhost:9092 \
#     --group alertas --describe
```

**Salida esperada**:
```
GROUP    TOPIC    PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
alertas  compras  0          15              20              5
alertas  compras  1          10              10              0
```

Un lag que crece indica que los consumers no dan abasto — necesitas más particiones o más consumers.

---

## 5. Kafka Connect y Kafka Streams

### Kafka Connect

Kafka Connect conecta Kafka con sistemas externos (bases de datos, S3, Elasticsearch) **sin escribir código**:

```
Source: PostgreSQL → Connect → Kafka Topic → Connect → Sink: S3
```

```bash
# Ejemplo: conector S3 sink
$ confluent connect create s3-sink \
    --config connector.class=io.confluent.connect.s3.S3SinkConnector \
    --config topics=compras \
    --config s3.bucket.name=mi-bucket \
    --config format.class=io.confluent.connect.s3.format.json.JsonFormat
```

### Kafka Streams

Kafka Streams (Java/Scala) permite transformaciones en el flujo sin necesidad de consumers/producers explícitos:

```java
KStream<String, Compra> compras = builder.stream("compras");

compras
    .filter((key, compra) -> compra.total > 10000)
    .to("alertas_compras_grandes");
```

Para Python existe `pykafka-streams` como alternativa, pero Kafka Streams nativo es más maduro. [[Time Series Anomaly Detection|Detección de anomalías en tiempo real]] es un caso de uso natural para Kafka Streams.

---

## 6. Kappa Architecture

La **Kappa Architecture** propone usar **streaming como fuente única de verdad**, eliminando la capa batch:

```
Eventos crudos → Kafka (log inmutable) → Stream Processor → Sinks
                                              ↓
                                       State stores / DBs
```

En lugar de tener datos "fríos" (batch) y "calientes" (streaming) separados, todo es un flujo continuo. Esto simplifica la arquitectura pero requiere madurez en streaming.

| | Lambda | Kappa |
|---|---|---|
| Capas | Batch + Streaming | Solo streaming |
| Complejidad | Alta (2 pipelines) | Media (1 pipeline) |
| Consistencia | Difícil (2 caminos) | Más fácil (1 camino) |
| Caso de uso | Cuando batch es inevitable | Nuevos sistemas greenfield |

[[Data Warehousing & Lakehouse|Data lakehouses]] modernos están adoptando cada vez más principios Kappa.

---

## 7. Common Mistakes

| Error | Explicación |
|---|---|
| Demasiadas particiones | Cada partición añade overhead. Empieza con 3-6 y escala según throughput. |
| Sin monitoreo de consumer lag | El lag crece silenciosamente hasta que los eventos se pierden. Monitorea siempre. |
| Sync commits en cada mensaje | `consumer.commit()` tras cada mensaje mata el throughput. Commitea cada N mensajes. |
| `acks=0` para datos críticos | Pierdes eventos si el broker falla. Usa `acks=all` para datos importantes. |
| Topics sin replicación | Si el broker muere, los datos se pierden. Usa `replication-factor >= 2` en producción. |

El monitoreo del consumer lag debe estar en tu dashboard desde el día 1. [[Data Quality & Testing|Tests de integración con Kafka]] ayudan a detectar problemas temprano.

---

## Resumen

1. **Streaming** procesa eventos en tiempo real (segundos), **batch** procesa lotes (minutos/horas).
2. Kafka tiene **topics** (categorías) divididos en **particiones** para escalar.
3. **Producers** envían eventos; **consumers** los leen en **consumer groups**.
4. `confluent-kafka` es la librería Python estándar para producers y consumers.
5. **Kafka Connect** integra fuentes/sinks externos sin código; **Kafka Streams** procesa flujos.
6. **Kappa Architecture** usa streaming como única fuente de verdad.
7. No monitorear consumer lag, tener demasiadas particiones, y usar sync commits son errores comunes.

---

## Comprueba tu Conocimiento

1. ¿Cuál es la diferencia entre un offset y una partición?
   <!-- Una partición es una división física del topic; el offset es el número secuencial de cada evento dentro de una partición. -->

2. ¿Por qué es peligroso `acks=0` en un producer?
   <!-- Porque no espera confirmación del broker — si el broker falla, el evento se pierde sin aviso. -->

3. ¿Qué es consumer lag y por qué monitorearlo?
   <!-- Es la diferencia entre el último evento producido y el último consumido. Un lag creciente indica que los consumers no dan abasto. -->

4. ¿Qué pasa si tienes 2 consumers en un grupo y un topic con 1 partición?
   <!-- Una partición solo la lee un consumer. El segundo consumer estará inactivo (sin asignación). -->

5. ¿Cuándo elegirías Kappa Architecture sobre Lambda?
   <!-- Cuando puedes permitirte un pipeline único de streaming, eliminando la complejidad de mantener dos pipelines (batch + streaming). -->

---

## ¿Dónde ir Siguente?

- [[Data Pipelines & ETL]] — complementa streaming con pipelines batch
- [[Workflow Orchestration (Airflow)]] — orquesta consumers que escriben a sinks
- [[Time Series Anomaly Detection]] — detecta anomalías en flujos de sensores
- [[Docker Fundamentals]] — levanta clusters Kafka con docker-compose
- [[Python for Data Science]] — patrones Python para producers y consumers robustos
