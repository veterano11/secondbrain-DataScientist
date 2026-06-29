---
tags: [software-engineering, ml-engineering, api, fastapi, python]
status: seedling
created: 2026-06-28
---

# API Design for ML

## 1. Escenario de aprendizaje

Tu modelo de clasificación de textos funciona perfectamente en Jupyter. Accuracy del 94%, matrices de confusión preciosas, todo bellamente documentado en celdas. Entonces llega el equipo de producto: quieren consumir el modelo desde la app mobile. Necesitas exponer el modelo como una API REST que reciba texto y devuelva predicciones, con validación, logging, versionado, y sin exponer los detalles internos.

Este salto de "funciona en mi máquina" a "funciona en producción" es donde muchos proyectos de ML fracasan. No porque el modelo sea malo, sino porque la API que lo sirve es frágil, no tiene validación, los errores son crípticos, y no hay forma de saber qué requests llegaron cuando algo sale mal.

En esta nota construirás una API para servir modelos de ML usando [[FastAPI]], el framework moderno de Python para APIs. No cubrimos deployment (eso es [[Model Serving]]), sino el diseño de la API en sí: cómo estructurar endpoints, validar datos, versionar, y monitorear.

## 2. Requisitos

- Python 3.9+
- FastAPI y uvicorn instalados (`pip install fastapi uvicorn pydantic`)
- Un modelo entrenado guardado con pickle o en formato ONNX
- Conocimientos básicos de [[Python for Data Science]] y HTTP

## 3. REST Fundamentals

REST (Representational State Transfer) es un estilo de arquitectura para APIs. En una API REST:

- **Recursos** son sustantivos (predictions, models, feedback)
- **Métodos HTTP** indican la acción: GET (leer), POST (crear), PUT (actualizar), DELETE (borrar)
- **Status codes** comunican el resultado: 200 (OK), 201 (creado), 400 (bad request), 404 (no encontrado), 500 (error interno)

Para servir un modelo de ML, el endpoint principal es POST /predict: el cliente envía datos y recibe predicciones. No uses GET para predicciones porque GET tiene límites de tamaño en la URL y no debe tener efectos secundarios (aunque la predicción en sí no modifica nada, el logging interno sí es un efecto).

```python
# Mini ejemplo de cómo NO hacerlo
from flask import Flask, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=["GET"])
def predict():
    data = request.args.get("data")  # Datos en URL — malo
    pred = model.predict([data])
    return {"prediction": pred[0]}
```

Problemas: datos en URL, sin validación, sin tipos, error 500 si algo falla sin mensaje claro.

## 4. FastAPI: Path Operations y Pydantic

FastAPI usa type hints de Python para validación automática, documentación interactiva (Swagger UI en /docs) y serialización. La clave está en los modelos de **pydantic** para request y response.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import List

app = FastAPI(title="ML Prediction API", version="1.0.0")
model = pickle.load(open("model.pkl", "rb"))

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., min_length=5, max_length=5,
        description="Vector de 5 features numéricas"
    )
    user_id: str = Field(None, description="ID del usuario para logging")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str = "1.0.0"

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X).max()
        return PredictionResponse(
            prediction=int(pred),
            probability=float(proba)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Pydantic valida automáticamente que `features` tenga exactamente 5 elementos numéricos. Si el cliente envía 3 features o un string, recibe un 422 con un mensaje descriptivo. Corre con `uvicorn app:app --reload` y visita `/docs` para ver la UI interactiva.

Salida esperada (curl):
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0], "user_id": "abc"}'
```

```json
{"prediction": 1, "probability": 0.87, "model_version": "1.0.0"}
```

## 5. Validación de Entrada

Uno de los errores más comunes en APIs de ML es asumir que los datos llegarán limpios. La validación debe ocurrir antes de que el modelo toque los datos. Pydantic permite validaciones complejas:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=5, max_length=5)
    user_id: Optional[str] = None

    @validator("features")
    def check_ranges(cls, v):
        for val in v:
            if val < 0 or val > 1:
                raise ValueError(
                    f"Feature {val} fuera de rango [0, 1]. "
                    "Normaliza los datos antes de enviarlos."
                )
        return v

    @validator("user_id")
    def check_user_id(cls, v):
        if v and len(v) > 100:
            raise ValueError("user_id demasiado largo (max 100 chars)")
        return v
```

```python
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "validation_error"}
    )
```

Salida esperada con datos inválidos:
```json
{"error": "Feature 2.5 fuera de rango [0, 1]. Normaliza los datos antes de enviarlos.", "type": "validation_error"}
```

Nunca confíes en que el cliente normalice los datos. Si tu modelo espera features en [0,1], valida que así sea en la API y devuelve un error claro antes de llegar al modelo. También puedes ofrecer un endpoint `/v1/transform` que haga la normalización del lado del servidor (útil para clients ligeros como apps mobile).

## 6. Versionado de APIs

Los modelos cambian. Cuando actualizas un modelo, los clients existentes no deben romperse. El versionado de APIs te permite mantener múltiples versiones simultáneamente.

Hay dos estrategias principales:

**URL-based versioning:** `/v1/predict`, `/v2/predict`
```python
@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_v1(request: PredictionRequest):
    # Modelo legacy
    pass

@app.post("/v2/predict", response_model=PredictionResponseV2)
async def predict_v2(request: PredictionRequestV2):
    # Modelo nuevo con más features
    pass
```

**Header-based versioning:** Accept header
```python
from fastapi import Header

@app.post("/predict")
async def predict(request: PredictionRequest, accept_version: str = Header("1.0")):
    if accept_version == "2.0":
        return await predict_v2(request)
    return await predict_v1(request)
```

URL-based es más simple y visible, header-based es más RESTful y limpio. Para proyectos ML, URL-based suele ser mejor porque es explícito en los logs y más fácil de depurar. Sea cual sea tu elección, implementa un plan de deprecation:

```python
import warnings

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_v1(request: PredictionRequest):
    warnings.warn("v1 deprecada, usa /v2/predict", DeprecationWarning)
    # ... lógica legacy ...
    response = PredictionResponse(prediction=pred, probability=proba)
    response.headers["X-API-Deprecated"] = "true"
    response.headers["X-API-Sunset"] = "2026-09-28"
    return response
```

Siempre comunica la deprecation con headers y con al menos 3 meses de ventana para migrar. Documenta los cambios de versión en el propio endpoint `/v1/docs` para que los consumidores sepan qué cambió.

## 7. Logging y Monitoreo

Cuando el modelo da una predicción incorrecta en producción, necesitas poder responder: ¿qué datos recibió? ¿cuánto tardó? ¿qué versión del modelo se usó? Sin logging, estás ciego.

```python
import logging
import time
from fastapi import Request
from datetime import datetime

logger = logging.getLogger("ml_api")
handler = logging.FileHandler("predictions.log")
handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
))
logger.addHandler(handler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    body = await request.body()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s "
        f"body={body.decode()[:500]}"
    )
    return response
```

Para monitoreo más estructurado, captura cada predicción individual con su contexto:

```python
from pydantic import BaseModel
from datetime import datetime

class PredictionLog(BaseModel):
    timestamp: datetime
    user_id: str
    features: list
    prediction: int
    probability: float
    latency_ms: float
    model_version: str

prediction_logs = []  # En producción: base de datos o servicio externo

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.time()
    # ... predicción ...
    latency = (time.time() - start) * 1000

    log_entry = PredictionLog(
        timestamp=datetime.utcnow(),
        user_id=request.user_id or "anonymous",
        features=request.features,
        prediction=int(pred),
        probability=float(proba),
        latency_ms=round(latency, 2),
        model_version="1.0.0"
    )
    prediction_logs.append(log_entry.dict())
    # Envía también a sistema de observabilidad (ver [[Observability]])

    response.headers["X-Latency-Ms"] = str(round(latency, 2))
    return PredictionResponse(prediction=int(pred), probability=float(proba))
```

Salida esperada en archivo de log:
```
2026-06-28 14:30:22,123 | INFO | path=/v1/predict status=200 duration=0.045s body={"features":[0.1,0.2,0.3,0.4,0.5],"user_id":"abc"}
```

Cada log tiene timestamp, latencia, input y output. Esto es oro para debugging: si un usuario reporta una predicción incorrecta, puedes buscar su user_id en los logs y reproducir el caso exacto.

## 8. Common Mistakes

- **Exponer el modelo sin validación:** Un string donde esperas un float no da un error bonito de Pydantic; da un 500 críptico o, peor, una predicción sin sentido.
- **No versionar la API:** Cambias el modelo, los clients mobile se rompen, nadie sabe por qué. Versionar es barato, no hacerlo es caro.
- **No rate-limit:** Un script malicioso (o un bug en el cliente) puede hacer 10,000 requests por segundo. Sin rate limiting, tu API cae y todos pierden. FastAPI no incluye rate limiting nativo, pero puedes usar slowapi o un proxy reverso como nginx.
- **Logging de datos sensibles:** Si el modelo recibe datos personales (nombres, emails, direcciones), no los loguees en texto plano. Anonimiza o tokeniza antes de logguear.
- **Endpoint único para todo:** /predict que hace de todo (predecir, reentrenar, borrar). Respeta REST: POST para crear/pedir, DELETE para borrar, cada endpoint con una responsabilidad.
- **No documentar el schema de entrada:** Tus clients no deberían adivinar qué enviar. FastAPI genera Swagger automáticamente, pero asegúrate de que los Field descriptions sean útiles y que el `/docs` sea accesible.
- **Ignorar errores del modelo:** Si el modelo tira una excepción, la API debe atraparla y devolver un 500 con un mensaje, no dejar que se propague como un error interno de FastAPI.

## Resumen

Diseñar una API para ML no es muy distinto de diseñar cualquier API, pero tiene matices importantes: validación estricta de datos de entrada (Pydantic), versionado explícito, logging estructurado de inputs/outputs/latencia, y rate limiting. FastAPI es el framework ideal para Python porque integra validación, documentación automática y async de manera nativa.

El objetivo final es que la API sea un contrato estable entre el modelo y sus consumidores. Los clients no deberían saber ni importarles cómo funciona el modelo internamente; solo necesitan enviar datos y recibir predicciones confiables.

## Check Your Understanding

1. ¿Por qué no deberías usar GET para un endpoint de predicción?
<!-- GET tiene límites de tamaño en URL y no debe producir efectos secundarios (logging es un efecto). Usa POST. -->

2. ¿Qué status code devuelve FastAPI automáticamente cuando falla la validación de Pydantic?
<!-- 422 Unprocessable Entity. -->

3. ¿Cuál es la ventaja de versionar por URL (/v1/, /v2/) vs por Header?
<!-- URL-based es más visible y fácil de depurar. Header-based es más RESTful. URL-based suele ser mejor para APIs de ML porque los logs muestran la versión explícitamente. -->

4. Menciona dos cosas que deberías loguear en cada request de predicción.
<!-- Timestamp, user_id, features, predicción, probabilidad, latencia, versión del modelo. -->

5. Si tu equipo decide no rate-limit la API, ¿qué podría pasar?
<!-- Un cliente bugueado o malicioso puede saturar el servidor con requests, degradando el servicio para todos los demás consumidores. -->

## Where to Go Next

- [[Design Patterns for Data Science]] — Los patrones de diseño que estructuran el código detrás de la API.
- [[FastAPI]] — Documentación oficial y avanzada del framework.
- [[Docker Fundamentals]] — Cómo containerizar tu API para deployment.
- [[Model Serving]] — Estrategias avanzadas de serving: batch, streaming, serverless.
- [[Observability]] — Métricas, tracing y alertas para APIs en producción.
- [[CLI & Productivity]] — Herramientas para testear tu API desde la terminal (httpie, curl scripts).
- [[Python for Data Science]] — Buenas prácticas de Python para el código de tu API.
