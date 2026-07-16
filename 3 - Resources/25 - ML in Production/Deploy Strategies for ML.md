---
tags:
  - ml-engineering
  - mlops
  - deploy-strategies
  - production-ml
status: seedling
created: 2026-06-28
---

# Deploy Strategies for ML

## 1. Escenario de aprendizaje

Eres ML Engineer en un e-commerce. Tu modelo de recomendaciones v1 genera el 35% de los ingresos. Desarrollaste v2 con un transformer que mejora el CTR 8% en offline. Pero si v2 falla en producción — respuestas lentas, malas recomendaciones — podrías perder millones.

No puedes reemplazar v1 por v2 directamente. Necesitas shadow deployment (validar sin impacto), canary release (escalar gradualmente), A/B testing (medir impacto en negocio), y rollback automático.

## 2. Requisitos

- Kubernetes básico, Docker, Python 3.10+
- Istio o Linkerd (opcional), `kubectl`, `helm`
- CI/CD básico (GitHub Actions), feature flags (LaunchDarkly o similar)

## 3. Shadow Deployment

Envía una copia del tráfico real al nuevo modelo, pero la respuesta nunca se muestra al usuario. Sirve para validar que no crashea con tráfico real y comparar predicciones offline.

```python
import asyncio, time, random, json
from dataclasses import dataclass

@dataclass
class PredictionResult:
    model_version: str
    prediction: any
    latency_ms: float
    error: str | None = None

class ShadowDeployer:
    def __init__(self, primary_model, shadow_model, log_path="shadow_logs.jsonl"):
        self.primary = primary_model
        self.shadow = shadow_model
        self.log_path = log_path

    async def predict(self, features: dict) -> PredictionResult:
        start = time.time()
        primary_result = self.primary.predict(features)
        primary_latency = (time.time() - start) * 1000

        try:
            shadow_result = self.shadow.predict(features)
            shadow_output = PredictionResult("v2-shadow", shadow_result, (time.time() - start) * 1000)
        except Exception as e:
            shadow_output = PredictionResult("v2-shadow", None, 0, str(e))

        with open(self.log_path, "a") as f:
            f.write(json.dumps(shadow_output.__dict__) + "\n")
        return PredictionResult("v1", primary_result, primary_latency)

class DummyModel:
    def __init__(self, version: str, fail_rate=0):
        self.version = version
        self.fail_rate = fail_rate
    def predict(self, features):
        if random.random() < self.fail_rate:
            raise ValueError("crash")
        return {"score": random.random(), "version": self.version}

v1, v2 = DummyModel("v1"), DummyModel("v2-shadow", fail_rate=0.05)
deployer = ShadowDeployer(v1, v2)
result = asyncio.run(deployer.predict({"user_id": 123}))
print(f"Shadow deploy: {result}")
```

### Shadow en Istio

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: recommender-shadow
spec:
  hosts:
  - recommender-service
  http:
  - route:
    - destination:
        host: recommender-service
        subset: v1
    mirror:
      host: recommender-service
      subset: v2
    mirrorPercentage:
      value: 100
```

## 4. Canary Release

Expone el nuevo modelo a un pequeño % de usuarios y aumenta gradualmente si las métricas son buenas.

```python
class CanaryRouter:
    def __init__(self, initial_percent=1.0):
        self.current_percent = initial_percent

    def should_route_to_canary(self, user_id: str) -> bool:
        return (hash(user_id) % 100) < self.current_percent

    def promote(self, new_percent: float):
        self.current_percent = min(new_percent, 100)
        print(f"Canary → {self.current_percent}%")

    def rollback(self):
        self.current_percent = 0
        print("Rollback: 100% a v1")

router = CanaryRouter(2.0)
router.promote(5.0)
router.promote(20.0)
router.promote(100.0)
```

**Salida esperada:** `Canary → 5.0% \n Canary → 20.0% \n Canary → 100.0%`

### Canary con Flagger

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: recommender-canary
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: recommender
  service:
    port: 8080
  analysis:
    interval: 1m
    stepWeight: 10
    maxWeight: 100
    metrics:
    - name: request-success-rate
      thresholdRange: { min: 99 }
      interval: 1m
    - name: request-duration
      thresholdRange: { max: 500 }
      interval: 1m
```

```bash
kubectl apply -f recommender-canary.yaml
kubectl get canaries -w
```

**Salida esperada:**
```
NAME                  STATUS        WEIGHT
recommender-canary    Progressing   10
recommender-canary    Progressing   50
recommender-canary    Promoting     100
```

## 5. A/B Testing de Modelos

Dos modelos compiten en vivo con asignación aleatoria. Decisión basada en métricas de negocio con significancia estadística.

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class ABTestResult:
    mean_a: float
    mean_b: float
    p_value: float
    significant: bool
    winner: str | None

class ABTest:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results_a, self.results_b = [], []

    def record(self, model: str, value: float):
        (self.results_a if model == "A" else self.results_b).append(value)

    def analyze(self) -> ABTestResult:
        t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
        mean_a, mean_b = np.mean(self.results_a), np.mean(self.results_b)
        significant = p_value < self.alpha
        return ABTestResult(mean_a, mean_b, p_value, significant,
                            "A" if mean_a > mean_b else "B" if significant else None)

test = ABTest()
for _ in range(10000):
    test.record("A", np.random.beta(10, 90))
    test.record("B", np.random.beta(12, 88))
result = test.analyze()
print(f"v1={result.mean_a:.4f}  v2={result.mean_b:.4f}  p={result.p_value:.4f}  winner={result.winner}")
```

**Salida esperada:** `v1=0.1002  v2=0.1198  p=0.0000  winner=B`

### Envoy Weighted Clusters

```yaml
route:
  weighted_clusters:
    clusters:
    - name: recommender-v1
      weight: 90
    - name: recommender-v2
      weight: 10
```

## 6. Model Versioning y Rollback

### Model Registry con MLflow

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
result = mlflow.register_model("runs:/abc123/model", "recommendation_transformer")
print(f"Registrado v{result.version}")

client.transition_model_version_stage("recommendation_transformer", 1, "Production")
```

**Salida esperada:** `Registrado v2`

### Rollback Automático

```python
class AutoRollback:
    def __init__(self, thresholds: dict, violation_limit=3):
        self.thresholds = thresholds
        self.violations = 0
        self.limit = violation_limit

    def check_metrics(self, metrics: dict) -> bool:
        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric)
            if threshold and value > threshold:
                print(f"[ALERT] {metric}: {value} > {threshold}")
                self.violations += 1
        if self.violations >= self.limit:
            print("Rollback ejecutado")
            return False
        return True

rollback = AutoRollback({"error_rate": 0.01, "latency_p99": 200})
for m in [{"error_rate": 0.05}, {"error_rate": 0.03}, {"latency_p99": 300}]:
    if not rollback.check_metrics(m):
        break
```

**Salida esperada:**
```
[ALERT] error_rate: 0.05 > 0.01
[ALERT] error_rate: 0.03 > 0.01
[ALERT] latency_p99: 300 > 200
Rollback ejecutado
```

## 7. Feature Flags para Modelos

Activar/desactivar modelos sin redeployar.

```python
class FeatureFlagClient:
    def __init__(self):
        self.flags = {"use_transformer_v2": False}

    def is_enabled(self, flag: str) -> bool:
        return self.flags.get(flag, False)

    def set_flag(self, flag: str, value: bool):
        self.flags[flag] = value
        print(f"{flag} → {value}")

flags = FeatureFlagClient()

class Recommender:
    def recommend(self, user_id: str) -> dict:
        if flags.is_enabled("use_transformer_v2"):
            return {"version": "v2", "score": 0.9}
        return {"version": "v1", "score": 0.7}

rec = Recommender()
print(rec.recommend("123"))
flags.set_flag("use_transformer_v2", True)
print(rec.recommend("123"))
```

**Salida esperada:** `{'version': 'v1', 'score': 0.7} \n use_transformer_v2 → True \n {'version': 'v2', 'score': 0.9}`

## 8. Common Mistakes

### Mistake 1: Shadow sin Monitoreo — nadie revisa los logs del shadow.
**Solución**: Dashboards de shadow con tasa de error, latencia, distribución de predicciones.

### Mistake 2: Canary sin Métricas Claras — subir de 1% a 100% sin definir éxito/fracaso.
**Solución**: Definir SLOs antes del deploy. Automatizar con Flagger.

### Mistake 3: No Tener Rollback Automático — si el canary falla a las 3 AM, nadie ejecuta `kubectl rollout undo`.
**Solución**: Auto-rollback basado en métricas. Flagger lo hace nativamente.

### Mistake 4: A/B Testing sin Significancia Estadística — ver CTR +0.5% y declarar victoria sin p-value.
**Solución**: Calcular tamaño de muestra antes del experimento. No detener temprano.

### Mistake 5: Mezclar Estrategias sin Control — shadow + canary + A/B al mismo tiempo sin sistema claro.
**Solución**: Separar concerns. Shadow = validación técnica, canary = rollout, A/B = experimentos de negocio.

## Resumen

- **Shadow deployment**: copia del tráfico al nuevo modelo sin afectar decisiones.
- **Canary release**: 1% → 5% → 50% → 100% con monitoreo continuo.
- **A/B testing**: dos modelos compiten; decisión con significancia estadística.
- **Model versioning**: MLflow registry + versionado semántico (MAJOR.MINOR.PATCH).
- **Rollback automático**: detectar degradación y revertir a versión estable.
- **Feature flags**: activar/desactivar modelos sin redeployar (kill switch).

## Comprueba tu Conocimiento

1. ¿Cuál es la diferencia entre shadow deployment y canary release?
   <!-- Shadow no afecta al usuario (copia del tráfico), canary sí afecta a un % de usuarios reales. -->
2. ¿Qué métricas monitorearías durante un canary?
   <!-- Error rate, latencia p50/p99, throughput, y métrica de negocio objetivo (CTR, conversión). -->
3. ¿Por qué no es suficiente promediar métricas en un A/B test?
   <!-- Necesitas significancia estadística (p-value) para saber si la diferencia no es por azar. -->
4. ¿Cómo implementarías un kill switch para un modelo errático?
   <!-- Feature flag que desvía tráfico a versión anterior. Monitoreo automático activa el flag si error_rate > umbral. -->
5. ¿Qué ventaja tiene Flagger sobre scripts caseros de canary?
   <!-- Análisis automatizado de métricas, progresión incremental, rollback automático. -->
6. ¿Cuándo usarías A/B testing en lugar de canary?
   <!-- Cuando necesitas un experimento estadístico riguroso con grupos de control y tratamiento. -->
7. ¿Por qué es importante el versionado semántico en modelos?
   <!-- Para saber si dos versiones son compatibles (mismo MAJOR) y decidir si requieren migración. -->
8. ¿Qué rol cumple Istio en estrategias de deploy de modelos?
   <!-- Service mesh que permite mirroring (shadow), weighted routing (canary), y header-based routing (A/B). -->

## ¿Dónde ir Siguente?

- [[Model Serving]] — infraestructura de servidores de modelos
- [[CI-CD & GitOps]] — automatizar pipelines de deploy y rollback
- [[A-B Testing]] — diseño de experimentos y estadística para ML
- [[Feature Stores]] — features consistentes entre versiones de modelos
- [[Model Monitoring]] — dashboards y alertas para detectar degradación
- [[Kubernetes Fundamentals]] — orquestación de contenedores para canary
- [[Docker Fundamentals]] — empaquetado de modelos para deploy reproducible
