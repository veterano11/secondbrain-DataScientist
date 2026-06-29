---
tags:
  - ml-engineering
  - mlops
  - model-serving
  - production-ml
status: seedling
created: 2026-06-28
---

# Model Serving

## 1. Escenario de aprendizaje

Tu equipo fine-tuneó BERT para clasificación de intenciones en un chatbot. En Jupyter predice en 300ms por sample. El producto necesita 1000 requests concurrentes con latencia p99 < 100ms. "Cargar un pickle y llamar a predict()" no escala.

Model serving implica infraestructura: servidores eficientes, batching dinámico, GPU scheduling, escalado horizontal, y versionado. Aprenderás a servir modelos con Triton, configurar batch y online inference, escalar, y evitar errores de latencia.

## 2. Requisitos

- Docker, Python 3.10+, PyTorch 2.0+, Triton Inference Server
- `locust` o `vegeta` para pruebas de carga
- Conceptos básicos de Kubernetes

## 3. Batch Inference

Cuando puedes esperar minutos u horas, el batch inference es más eficiente: recomendaciones diarias, scoring de leads, generación de embeddings.

```python
class BatchPredictor:
    def __init__(self, model_path: str):
        self.model = lambda X: np.random.rand(len(X), 2)
        self.batch_size = 32

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        results = []
        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i + self.batch_size]
            results.append(self.model(batch.values))
        return np.concatenate(results, axis=0)

import pandas as pd, numpy as np, time
data = pd.DataFrame(np.random.randn(10000, 128))
predictor = BatchPredictor("models/bert_v1.pt")
start = time.time()
preds = predictor.predict_batch(data)
elapsed = time.time() - start
print(f"{len(preds)} preds en {elapsed:.2f}s — {len(preds)/elapsed:.0f} samples/s")
```

**Salida esperada:** `10000 preds en 0.42s — 23809 samples/s`

## 4. Online Inference: REST y gRPC

### REST API con FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np, time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    latency_ms: float

class ModelServer:
    def __init__(self):
        self.model = lambda text: np.random.rand(4)
        self.labels = ["cancel_order", "track_order", "return_item", "contact_support"]

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.model(text)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

server = ModelServer()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.time()
    intent, confidence = server.predict(request.text)
    latency = (time.time() - start) * 1000
    logger.info(f"{intent} ({confidence:.2f}) en {latency:.1f}ms")
    return PredictionResponse(intent=intent, confidence=confidence, latency_ms=round(latency, 1))
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Quiero cancelar mi pedido"}'
```

**Salida esperada:**
```json
{"intent":"cancel_order","confidence":0.94,"latency_ms":85.3}
```

### gRPC vs REST

gRPC usa serialización binaria (Protobuf) y es 2-5x más rápido que REST en alto throughput.

```python
import grpc
import model_pb2_grpc

class GRPCClient:
    def __init__(self, host="localhost:8500"):
        channel = grpc.insecure_channel(host)
        self.stub = model_pb2_grpc.InferenceServiceStub(channel)

    def predict(self, text: str) -> tuple[str, float]:
        response = self.stub.Predict(model_pb2.InferenceRequest(text=text))
        return response.intent, response.confidence
```

## 5. NVIDIA Triton Inference Server

Triton es el estándar industrial para servir modelos. Soporta múltiples frameworks, batching dinámico, y GPU/CPU scheduling.

### Configuración del Modelo

```
models/bert_intents/1/model.onnx
models/bert_intents/config.pbtxt
```

```protobuf
name: "bert_intents"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [{ name: "input_ids", data_type: TYPE_INT64, dims: [128] }]
output [{ name: "probs", data_type: TYPE_FP32, dims: [4] }]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100
}
instance_group [{ count: 2, kind: KIND_GPU }]
```

### Desplegar Triton

```bash
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

**Salida esperada:**
```
+----------------------------+---------+--------+
| Model                       | Version | Status |
+----------------------------+---------+--------+
| bert_intents                | 1       | READY  |
+----------------------------+---------+--------+
```

### Cliente Python

```python
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
inputs = [httpclient.InferInput("input_ids", [1, 128], "INT64")]
inputs[0].set_data_from_numpy(input_ids.numpy())
outputs = [httpclient.InferRequestedOutput("probs")]

response = client.infer("bert_intents", inputs, outputs=outputs)
probs = response.as_numpy("probs")
print(probs[0])
```

**Salida esperada:** `[0.02 0.91 0.04 0.03]`

### Dynamic Batching

Triton acumula requests hasta completar un batch óptimo, mejorando throughput sin sacrificar latencia.

```python
import threading, time

results = []
def send():
    start = time.time()
    client.infer("bert_intents", inputs)
    results.append((time.time() - start) * 1000)

threads = [threading.Thread(target=send) for _ in range(50)]
for t in threads: t.start()
for t in threads: t.join()

latencies = sorted(results)
p50 = latencies[len(latencies)//2]
p99 = latencies[int(len(latencies)*0.99)]
print(f"p50: {p50:.1f}ms, p99: {p99:.1f}ms")
```

**Salida esperada:** `p50: 45.2ms, p99: 92.8ms`

## 6. Escalado

### Horizontal (Kubernetes HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bert-intents-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: triton_request_latency_p99
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
```

### Cold Start y Warming

El primer request post-deploy tarda 10x más porque carga pesos y compila grafos.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up...")
    for text in ["Quiero cancelar", "¿Dónde está mi paquete?", "Necesito devolver"]:
        server.predict(text)
    print("Listo")
    yield

app = FastAPI(lifespan=lifespan)
```

## 7. Model Versioning

### Canary en Triton

```bash
curl -X POST localhost:8000/v2/repository/models/bert_intents/versions/2/load
curl -X POST localhost:8000/v2/repository/models/bert_intents/config \
  -H "Content-Type: application/json" \
  -d '{"version_policy": {"specific": {"versions": [1, 2]}}}'
```

### A/B Router

```python
import random

class ABRouter:
    def __init__(self, model_a: str, model_b: str, split: float = 0.1):
        self.model_a, self.model_b, self.split = model_a, model_b, split

    def route(self) -> str:
        return self.model_b if random.random() < self.split else self.model_a

router = ABRouter("bert_intents:v1", "bert_intents:v2")
print(f"Routing to: {router.route()}")
```

## 8. Common Mistakes

### Mistake 1: No Medir Latencia p99 — el promedio oculta colas largas.
**Solución**: Monitorear p50, p95, p99. Alertar cuando p99 supere el SLO.

### Mistake 2: Modelos sin Warmup — el primer request puede tardar 10x más.
**Solución**: Enviar requests sintéticos inmediatamente después del deploy.

### Mistake 3: Batching sin Timeout — si el servidor espera indefinidamente a llenar un batch, los primeros requests quedan atrapados.
**Solución**: Configurar `max_queue_delay_microseconds` en Triton.

### Mistake 4: Ignorar el GIL en Python — un modelo PyTorch en CPU no escala con threads.
**Solución**: Usar Triton (ejecuta en C++) o multiprocessing (cada worker su propio proceso).

### Mistake 5: No Versionar Endpoints — clientes antiguos reciben predicciones incompatibles.
**Solución**: Incluir versión en la ruta (`/v1/predict`, `/v2/predict`).

## Resumen

- **Batch inference** para cargas programadas (Spark, schedulers); **online inference** para APIs con latencia <100ms.
- **Triton** maneja múltiples frameworks, dynamic batching, y scheduling GPU/CPU.
- **Dynamic batching** agrupa requests concurrentes; configurar timeout para evitar starvation.
- **Escalado horizontal** con HPA en K8s; **warmup** obligatorio para evitar cold starts.
- Monitorear **p99 de latencia** — no promedio.

## Check Your Understanding

1. ¿Por qué el promedio de latencia no es suficiente para monitorear model serving?
   <!-- El promedio oculta colas largas; el p99 revela la experiencia del percentil más lento. -->
2. ¿Qué ventaja tiene gRPC sobre REST para inferencia?
   <!-- Serialización binaria, streaming bidireccional, 2-5x menos latencia en alto throughput. -->
3. ¿Cómo funciona el dynamic batching en Triton?
   <!-- Acumula requests hasta alcanzar un batch preferido o vencer un timeout, luego ejecuta el batch completo. -->
4. ¿Qué métrica usarías para escalar horizontalmente un servidor de modelos?
   <!-- Latencia p99, utilización de CPU/GPU, o requests por segundo. -->
5. ¿Por qué un modelo PyTorch en CPU no escala con threading en Python?
   <!-- El GIL impide ejecución paralela de threads; usar multiprocessing o Triton. -->
6. ¿Qué es un cold start y cómo se mitiga?
   <!-- Primer request lento porque el modelo no está en memoria; mitigar con warmup al arrancar. -->
7. ¿Cómo implementarías un canary deployment de un nuevo modelo?
   <!-- Enrutar 5-10% del tráfico al nuevo modelo, monitorear, aumentar gradualmente. -->
8. ¿Qué configuración de Triton evita que un request espere demasiado en el batch?
   <!-- max_queue_delay_microseconds — tiempo máximo antes de ejecutar el batch. -->

## Where to Go Next

- [[API Design for ML]] — mejores prácticas para APIs de inferencia
- [[Docker Fundamentals]] — contenedorización de modelos para deploy
- [[Kubernetes Fundamentals]] — orquestación de servidores de modelos
- [[Deploy Strategies for ML]] — shadow, canary, y A/B testing de modelos
- [[Model Compression]] — cuantizar y podar modelos para reducir latencia
- [[Continuous Training]] — reentrenar y validar modelos automáticamente
- [[Observability]] — monitoreo de latencia, throughput y errores en serving
