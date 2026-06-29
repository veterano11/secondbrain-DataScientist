---
tags: [software-engineering, system-design, ml-engineering, scalability]
status: seedling
created: 2026-06-28
---

# System Design for ML

## 1. Escenario de aprendizaje

Tu startup de e-commerce quiere recomendar productos en tiempo real. Tienes 10 millones de usuarios activos, un catálogo de 500,000 productos, y cada usuario debe recibir recomendaciones personalizadas en menos de 200ms. El modelo de recomendaciones es solo una pieza del rompecabezas: necesitas un feature store que sirva embeddings actualizados, una capa de caché para usuarios frecuentes, una cola de mensajes para ingerir eventos de compra en tiempo real, y un load balancer que distribuya el tráfico entre réplicas del modelo.

El modelo sin el sistema alrededor es un experimento de Jupyter bonito pero inservible. **System design** es el arte de poner todas las piezas juntas para que el sistema funcione bajo carga, sea resiliente a fallos, y pueda escalar cuando lleguen 10 millones más de usuarios.

Esta nota asume que conoces los fundamentos de APIs, bases de datos y redes. Si alguno de esos conceptos te resulta nuevo, revisa [[API Design for ML]] y [[Data Engineering]] primero.

## 2. Requisitos

- Conceptos básicos de redes (TCP/IP, HTTP, DNS)
- Familiaridad con REST APIs y bases de datos (SQL y NoSQL)
- Haber trabajado con datos a escala de miles de registros (para apreciar por qué las cosas cambian a millones)
- Ganas de dibujar arquitecturas en una pizarra virtual

## 3. Componentes de un Sistema ML

Un sistema de recomendaciones en producción tiene estos componentes principales:

**Feature Store:** Repositorio centralizado de features precalculadas. En lugar de que cada servicio calcule "última compra del usuario" cada vez, el feature store mantiene features actualizadas y las sirve con baja latencia. Es una base de datos clave-valor (como [[Feature Stores]] de Feast o Tecton) optimizada para lectura rápida.

**Model Server:** El modelo entrenado expuesto como servicio. Puede ser un contenedor Docker con FastAPI sirviendo predicciones, o un servicio especializado como TensorFlow Serving o TorchServe.

**Cache:** Almacenamiento temporal de predicciones frecuentes. Si el usuario A pide recomendaciones cada 5 minutos, no necesitas ejecutar el modelo cada vez; cachea el resultado y expíralo cuando haya nueva data.

**Load Balancer:** Distribuye requests entre múltiples réplicas del model server. Sin load balancer, un pico de tráfico en una réplica la satura mientras otras están ociosas.

**Message Queue:** Cola de eventos asíncrona. Cuando un usuario compra un producto, no esperes a que el modelo reaccione inmediatamente. Publica el evento en Kafka o RabbitMQ, y un consumidor lo procesa cuando pueda.

**API Gateway:** Punto de entrada único que maneja autenticación, rate limiting, y enrutamiento a los servicios internos.

```python
# Diagrama conceptual (mental: dibújalo como cajas y flechas)
#
# [Mobile App] -> [API Gateway] -> [Load Balancer] -> [Model Server] -> [Feature Store]
#                                                      -> [Cache (Redis)]
# [Event Bus (Kafka)] -> [Feature Updater] -> [Feature Store]
```

## 4. Sincrónico vs Asincrónico

Una decisión clave es si el cliente espera la respuesta (sincrónico) o no (asincrónico).

**Sincrónico (REST):** El cliente envía la request y espera. Es simple, directo, y funciona bien cuando la predicción tarda menos de 200ms. El problema: si el feature store está lento, la request se bloquea.

```python
# Pseudocódigo de un endpoint sincrónico
@app.post("/v1/recommend")
async def recommend(user_id: str):
    features = await feature_store.get(user_id)  # Espera bloqueante
    recs = await model.predict(features)         # Espera bloqueante
    return {"recommendations": recs}
```

**Asincrónico (Message Queue):** El cliente publica un mensaje y recibe un ID de tracking. Después consulta el resultado. Útil cuando la predicción es costosa (resumen de video, recomendación semanal) o cuando el sistema está bajo alta carga.

```python
# Pseudocódigo de un endpoint asincrónico
@app.post("/v1/recommend/async")
async def recommend_async(user_id: str):
    job_id = str(uuid.uuid4())
    kafka_producer.send("recommendation_jobs", {"job_id": job_id, "user_id": user_id})
    return {"job_id": job_id}

@app.get("/v1/recommend/async/{job_id}")
async def get_result(job_id: str):
    result = redis.get(f"job:{job_id}")
    if result:
        return json.loads(result)
    return {"status": "processing"}
```

**Message Queues** como [[Streaming & Event-Driven (Kafka)]] son el estándar para sistemas asincrónicos. Kafka particularmente es ideal por su durabilidad, particionado y capacidad de replay (reprocesar eventos si algo falla).

Para el escenario de recomendaciones en tiempo real, usa un modelo híbrido: el endpoint principal es sincrónico para requests normales, pero eventos de compra se publican en Kafka para actualizar el feature store asincrónicamente.

## 5. Caching — Redis, CDN, Cache Policy

El cache es la herramienta más efectiva para reducir latencia y carga en el modelo. No todas las predicciones necesitan ser frescas.

**Estrategias de cache:**

- **TTL (Time-To-Live):** La entrada expira después de N segundos. Para recomendaciones de e-commerce, un TTL de 5-15 minutos suele ser aceptable.
- **LRU (Least Recently Used):** Cuando el cache está lleno, elimina los elementos menos usados. Redis lo implementa nativamente con `maxmemory-policy allkeys-lru`.
- **Cache warming:** Precarga el cache con usuarios populares antes de un pico de tráfico (Black Friday, Cyber Monday).

```python
import redis
import json

cache = redis.Redis(host="localhost", port=6379, decode_responses=True)

async def get_recommendations(user_id: str) -> dict:
    cached = cache.get(f"recs:{user_id}")
    if cached:
        return json.loads(cached)

    features = await feature_store.get(user_id)
    recs = await model.predict(features)

    cache.setex(f"recs:{user_id}", 300, json.dumps(recs))  # TTL = 5 min
    return recs
```

Salida esperada:
```python
# Primera llamada: miss en cache (consulta a feature store + modelo)
# Segunda llamada (5 min): hit en cache (< 1ms)
```

También considera una **CDN** para assets estáticos (no aplica a predicciones, pero sí a las imágenes de productos que acompañan las recomendaciones). Redis es el rey del caching en sistemas ML por su velocidad (sub-milisegundo) y flexibilidad.

## 6. Escalado — Horizontal vs Vertical

Cuando 10M usuarios se convierten en 50M, tu sistema debe escalar.

**Vertical scaling (scale up):** Máquina más grande. Más RAM, más CPUs, GPU más potente. Simple, pero tiene límites (la máquina más cara de AWS tiene ~24TB de RAM, y cuesta una fortuna). Además, hay un solo punto de fallo.

**Horizontal scaling (scale out):** Más máquinas pequeñas. Es la estrategia de internet: en lugar de un servidor monstruo, 100 servidores normales con un load balancer al frente.

Para ML, el cuello de botella suele ser la inferencia. Si el modelo es ligero (regresión logística, random forest pequeño), escala horizontalmente con facilidad. Si el modelo es una red neuronal grande (BERT, GPT), necesitarás GPUs y quizás escalado vertical con una sola réplica poderosa.

**Auto-scaling:** El sistema añade o quita réplicas automáticamente según la carga. Kubernetes ([Kubernetes Fundamentals]) lo maneja con HorizontalPodAutoscaler basado en CPU, memoria, o métricas personalizadas (QPS, latencia).

```yaml
# Ejemplo de auto-scaling en Kubernetes
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Sharding de features:** Cuando el feature store no cabe en una máquina, particiona los datos por usuario (`user_id % N_SHARDS`). Cada shard es independiente, lo que permite escalar horizontalmente también el feature store.

## 7. Estimación de Capacidad

Antes de construir, estima cuántos recursos necesitas. Las preguntas clave:

**QPS (Queries Per Second):** ¿Cuántas predicciones por segundo necesitas?
- 10M usuarios, cada uno hace 5 requests/día = 50M requests/día
- 50M / 86400 segundos ≈ 579 QPS en promedio
- Pico (2x promedio): ~1200 QPS

**Latencia:** ¿Cuánto tarda cada componente?
- Feature store: 5-20ms (Redis, en memoria)
- Modelo: 10-100ms (según complejidad)
- Red: 10-50ms (entre servicios en misma región)
- Total: 25-170ms (dentro del target de 200ms)

**Throughput:** ¿Cuántas requests puede manejar una réplica?
- Si cada request toma 50ms, una réplica maneja 20 requests/segundo
- Para 1200 QPS pico: 1200 / 20 = 60 réplicas mínimo (con margen: 75)

**Costo por inferencia:**
- 50M requests/día * 30 días = 1,500M requests/mes
- Si cada inferencia cuesta $0.0001 (cálculo estimado en CPU): $150,000/mes solo en inferencia
- El feature store y cache añaden costo de infraestructura (Redis, máquinas)

```python
def estimate_cost(qps: int, latency_ms: int, cost_per_hour: float):
    replicas = (qps * latency_ms / 1000) * 1.5  # 1.5x margen de seguridad
    monthly_cost = replicas * cost_per_hour * 730  # 730 horas/mes
    return replicas, monthly_cost

r, c = estimate_cost(1200, 50, 0.50)
print(f"Replicas estimadas: {r:.0f}, Costo mensual: ${c:.0f}")
```

Salida esperada:
```
Replicas estimadas: 90, Costo mensual: $32850
```

Estas estimaciones son aproximadas pero te dan una base para la discusión con el equipo de infraestructura.

## 8. Common Mistakes

- **Ignorar latencia de red:** Suponer que todos los servicios responden en 0ms. En la práctica, cada salto de red añade 5-50ms. Cuando tu sistema tiene 5 saltos (API Gateway -> Load Balancer -> Model Server -> Feature Store -> Cache), acumulas 100-250ms solo en red.
- **No planificar picos de tráfico:** El sistema funciona en hora valle pero colapsa en promociones. Si no tienes auto-scaling y pruebas de carga, descubrirás los límites en el peor momento.
- **Cache sin invalidación:** El cache perfecto tiene datos frescos. Si cacheas recomendaciones por 1 hora pero el usuario compra algo a los 5 minutos, las recomendaciones están desactualizadas. Usa TTLs cortos o invalida el cache cuando ocurren eventos de compra.
- **Feature store como base de datos relacional:** Usar PostgreSQL para features en tiempo real es lento. Los feature stores deben ser key-value stores (Redis, Cassandra, DynamoDB) optimizados para lectura por clave.
- **Monolito de inferencia:** El mismo servicio que entrena también predice. Separa training (batch, pesado, ocasional) de serving (tiempo real, ligero, constante).
- **No estimar capacidad antes de construir:** Llegar a producción y descubrir que necesitas 10x más recursos es catastrófico. Haz estimaciones gruesas al principio y refínalas con el tiempo.

## Resumen

Un sistema de ML en producción es mucho más que el modelo. El feature store, cache, message queue, load balancer, y auto-scaling son tan importantes como la accuracy del modelo. Las decisiones clave son: sincrónico vs asincrónico (depende de la latencia tolerable), qué cachear y por cuánto tiempo (TTL vs LRU), y cómo escalar (horizontal vs vertical).

System design no es una receta fija; cada escenario tiene restricciones distintas. Pero los componentes y patrones se repiten: cache, colas, feature stores, load balancers. Reconocer el patrón correcto para tu problema es la habilidad principal.

## Check Your Understanding

1. ¿Cuándo preferirías un endpoint asincrónico sobre uno sincrónico para servir predicciones?
<!-- Cuando la predicción es costosa (varios segundos), cuando el sistema está bajo alta carga, o cuando el cliente no necesita la respuesta inmediatamente. -->

2. ¿Qué problema resuelve un feature store que una base de datos relacional no resuelve bien?
<!-- Bajísima latencia de lectura (< 10ms) por clave, optimizado para servir features precalculadas en tiempo real, no para consultas ad-hoc. -->

3. Si cacheas recomendaciones con TTL de 30 minutos, ¿qué pasa si un usuario cambia sus preferencias?
<!-- Las recomendaciones seguirán siendo las viejas hasta que el TTL expire. Para mitigar, invalida el cache manualmente cuando el usuario actualiza preferencias. -->

4. ¿Cuántas réplicas necesitas para manejar 500 QPS si cada request tarda 100ms?
<!-- Cada réplica maneja 10 requests/segundo (1000ms / 100ms). 500 / 10 = 50 réplicas, más margen de seguridad (~75). -->

5. ¿Por qué el escalado vertical tiene límites más allá del costo?
<!-- Porque incluso la máquina más grande del mundo tiene límites físicos de RAM, CPU y GPU. Además, crea un solo punto de fallo. -->

## Where to Go Next

- [[API Design for ML]] — Cómo diseñar los endpoints que conectan tu sistema con los clients.
- [[Feature Stores]] — Profundiza en el diseño y operación de feature stores.
- [[Model Serving]] — Estrategias avanzadas: TensorFlow Serving, TorchServe, ONNX Runtime.
- [[Kubernetes Fundamentals]] — Orquestación de contenedores para escalar tu sistema.
- [[Streaming & Event-Driven (Kafka)]] — Cómo manejar eventos en tiempo real con Kafka.
- [[Data Engineering]] — Pipelines de datos que alimentan tu feature store y entrenamiento.
- [[CLI & Productivity]] — Herramientas para testear carga y monitorear tu sistema desde la terminal.
