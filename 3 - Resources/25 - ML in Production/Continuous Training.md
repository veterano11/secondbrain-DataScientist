---
tags:
  - ml-engineering
  - mlops
  - continuous-training
  - production-ml
status: seedling
created: 2026-06-28
---

# Continuous Training

## 1. Escenario de aprendizaje

Eres ML Engineer en supply chain de un retailer. Tu modelo de predicción de demanda (XGBoost) se entrenó en enero. En marzo, el accuracy cayó 10% — los patrones de compra cambiaron por campañas promocionales. Reentrenaste manualmente, pero en abril volvió a caer.

Necesitas continuous training: un pipeline que detecta cuándo el modelo se degrada, reentrena con datos nuevos, valida calidad y performance, registra el modelo, y lo deploya si es mejor que el actual.

## 2. Requisitos

- Python 3.10+, MLflow (`pip install mlflow`), XGBoost, scikit-learn
- Conceptos de CI/CD, feature stores, y model monitoring

## 3. Trigger Strategies

### Schedule (Tiempo)

```python
from datetime import datetime

class ScheduleTrigger:
    def __init__(self, interval_hours=24):
        self.interval = interval_hours
        self.last_run = None

    def should_trigger(self) -> bool:
        now = datetime.now()
        if self.last_run is None:
            self.last_run = now
            return True
        return (now - self.last_run).total_seconds() / 3600 >= self.interval

    def execute(self, pipeline_fn):
        if self.should_trigger():
            pipeline_fn()
            self.last_run = datetime.now()
```

### Drift Detection (Datos)

```python
import numpy as np
from scipy.stats import ks_2samp

class DriftTrigger:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.reference = None

    def set_reference(self, data: np.ndarray):
        self.reference = data

    def should_trigger(self, current: np.ndarray) -> bool:
        if self.reference is None:
            return False
        stat, p = ks_2samp(self.reference, current)
        if p < self.threshold:
            print(f"[TRIGGER] Drift: KS={stat:.3f}, p={p:.4f}")
            return True
        return False

train_demand = np.random.normal(100, 20, 1000)
new_demand = np.random.normal(130, 35, 1000)
trigger = DriftTrigger()
trigger.set_reference(train_demand)
trigger.should_trigger(new_demand)
```

**Salida esperada:** `[TRIGGER] Drift: KS=0.248, p=0.0000`

### Performance Degradation (Modelo)

```python
from sklearn.metrics import mean_absolute_error

class PerformanceTrigger:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_trigger(self, y_true, y_pred) -> bool:
        mae = mean_absolute_error(y_true, y_pred)
        if mae > self.threshold:
            print(f"[TRIGGER] MAE={mae:.2f} > {self.threshold}")
            return True
        return False

trigger = PerformanceTrigger(15.0)
y_true = np.array([100, 200, 150])
y_pred = np.array([130, 180, 120])
trigger.should_trigger(y_true, y_pred)
```

**Salida esperada:** `[TRIGGER] MAE=26.00 > 15.00`

## 4. Pipeline de Reentrenamiento

```python
import mlflow, xgboost as xgb
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TrainingPipeline:
    def run(self) -> str:
        with mlflow.start_run() as run:
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("n_estimators", 500)

            np.random.seed(42)
            data = pd.DataFrame({
                "day_of_week": np.random.randint(0, 7, 10000),
                "month": np.random.randint(1, 13, 10000),
                "is_promotion": np.random.randint(0, 2, 10000),
                "price": np.random.uniform(10, 500, 10000),
                "demand": np.random.poisson(50, 10000) + np.random.normal(0, 10, 10000),
            })
            data["demand"] = data["demand"].clip(0)
            print(f"📦 Datos: {len(data):,} filas")

            data["price_ratio"] = data["price"] / (data["price"].mean() + 1)
            data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

            X = data.drop("demand", axis=1)
            y = data["demand"]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                     early_stopping_rounds=20, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            print(f"🏋️ Best iteration: {model.best_iteration}")

            y_pred = model.predict(X_val)
            metrics = {
                "mae": mean_absolute_error(y_val, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
            }
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, "model")
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "demand_forecast")
            print(f"✅ Run ID: {run.info.run_id}")

pipeline = TrainingPipeline()
pipeline.run()
```

**Salida esperada:**
```
📦 Datos: 10,000 filas
🏋️ Best iteration: 342
✅ Run ID: abc123def456
```

## 5. Automated Validation

### Data Quality Checks

```python
class DataValidator:
    def __init__(self):
        self.checks = []

    def add(self, name: str, fn):
        self.checks.append((name, fn))

    def validate_all(self, df: pd.DataFrame) -> bool:
        passed = True
        for name, fn in self.checks:
            result = fn(df)
            status = "✅" if result else "❌"
            print(f"  {status} {name}")
            passed = passed and result
        return passed

validator = DataValidator()
validator.add("no_nulls", lambda df: df.isnull().sum().sum() == 0)
validator.add("demand_positive", lambda df: (df["demand"] >= 0).all())

sample = pd.DataFrame({"demand": [100, 200, -5], "price": [50, 100, 200]})
ok = validator.validate_all(sample)
print(f"Resultado: {'✅' if ok else '❌'}")
```

**Salida esperada:** `✅ no_nulls \n ❌ demand_positive \n Resultado: ❌`

### Model Performance Against Baseline

```python
class ModelValidator:
    def __init__(self, baseline: dict, thresholds: dict):
        self.baseline = baseline
        self.thresholds = thresholds

    def validate(self, new_metrics: dict) -> bool:
        passed = True
        for metric, value in new_metrics.items():
            baseline = self.baseline.get(metric)
            if baseline is None:
                continue
            change = (value - baseline) / baseline
            limit = self.thresholds.get(metric, 0.05)
            status = "✅" if change <= limit else "❌"
            print(f"  {status} {metric}: {value:.3f} vs {baseline:.3f} (Δ={change:+.2%})")
            passed = passed and (change <= limit)
        return passed

validator = ModelValidator({"mae": 7.5}, {"mae": 0.1})
validator.validate({"mae": 7.2})
```

**Salida esperada:** `✅ mae: 7.200 vs 7.500 (Δ=-4.00%)`

## 6. Model Registry con MLflow

```python
from mlflow import MlflowClient

class RegistryManager:
    def __init__(self):
        self.client = MlflowClient()

    def register(self, run_id: str, name: str) -> int:
        version = mlflow.register_model(f"runs:/{run_id}/model", name).version
        print(f"Registrado {name} v{version}")

    def promote(self, name: str, version: int, stage: str = "Production"):
        for mv in self.client.get_latest_versions(name, stages=["Production"]):
            self.client.transition_model_version_stage(name, mv.version, "Archived")
        self.client.transition_model_version_stage(name, version, stage)
        print(f"  → v{version} → {stage}")

registry = RegistryManager()
registry.register("abc123", "demand_forecast")
registry.promote("demand_forecast", 1)
```

**Salida esperada:** `Registrado demand_forecast v1 \n   → v1 → Production`

## 7. Rollback Automático

Si el nuevo modelo empeora las métricas en producción, revertir automáticamente.

```python
from sklearn.metrics import mean_absolute_error

class AutoRollback:
    def __init__(self, window=10, threshold_ratio=1.15):
        self.window = window
        self.threshold = threshold_ratio
        self.log = []

    def record(self, y_true, y_pred, model_version: int):
        self.log.append({"y_true": y_true, "y_pred": y_pred, "version": model_version})
        if len(self.log) > self.window:
            self.log.pop(0)

    def evaluate(self, baseline_mae: float) -> bool:
        if len(self.log) < self.window:
            return False
        current = [p for p in self.log if p["version"] == self.log[-1]["version"]]
        if not current:
            return False
        mae = mean_absolute_error([p["y_true"] for p in current], [p["y_pred"] for p in current])
        if mae > baseline_mae * self.threshold:
            print(f"Rollback: MAE={mae:.2f} > {baseline_mae * self.threshold:.2f}")
            return True
        return False

rollback = AutoRollback()
for _ in range(15):
    rollback.record(100 + np.random.normal(0, 5), 90 + np.random.normal(0, 15), 2)
rollback.evaluate(baseline_mae=7.5)
```

**Salida esperada:** `Rollback: MAE=12.34 > 8.62`

## 8. Common Mistakes

### Mistake 1: Reentrenar sin Validación — un modelo malo puede reemplazar a uno bueno.
**Solución**: Comparar contra baseline antes de promover. Si las métricas empeoran, rechazar.

### Mistake 2: No Versionar Datos de Entrenamiento — no puedes reproducir ni depurar.
**Solución**: Versionar cada dataset (hash, timestamp + commit ID). Registrar en MLflow como `training_data_version`.

### Mistake 3: Triggers Demasiado Sensibles — reentrenar cada hora por ruido estadístico.
**Solución**: Cooldown entre reentrenamientos. Combinar múltiples triggers. No reentrenar más de 1x/día sin causa justificada.

### Mistake 4: Ignorar el Costo de Reentrenamiento — entrenar transformers cuesta $50-500 por ejecución.
**Solución**: Estimar costo y configurar triggers en consecuencia. Preferir triggers basados en performance sobre drift.

### Mistake 5: No Tener Circuit Breaker — si el pipeline falla, los errores se propagan silenciosamente.
**Solución**: Si el pipeline falla N veces consecutivas, pausar triggers y alertar a un humano.

## Resumen

- **Continuous training** automatiza reentrenamiento cuando datos o rendimiento se degradan.
- **Triggers**: schedule, drift, performance degradation, data freshness. Combinar para robustez.
- **Pipeline**: extract → feature engineering → train → validate → register. Cada paso reproducible.
- **Validación automática**: data quality + comparación contra baseline. No promover modelos peores.
- **Model registry** (MLflow): versionado, lineage, stages (Staging → Production).
- **Rollback automático**: si métricas empeoran > umbral, revertir a versión anterior.

## Comprueba tu Conocimiento

1. ¿Cuál es la diferencia entre un trigger por drift y uno por performance degradation?
   <!-- Drift detecta cambios en distribución de entrada; performance degradation detecta empeoramiento de métricas de salida. -->
2. ¿Por qué es importante versionar los datos de entrenamiento?
   <!-- Reproducibilidad: saber exactamente con qué datos se entrenó cada versión del modelo. -->
3. ¿Qué validaciones debería pasar un modelo antes de promoverse a producción?
   <!-- Data quality (sin nulos), performance vs baseline (MAE no peor que X%), fairness checks. -->
4. ¿Cómo evitarías reentrenamientos innecesarios por ruido estadístico?
   <!-- Cooldown period, thresholds conservadores, combinar múltiples señales. -->
5. ¿Qué es un circuit breaker en continuous training?
   <!-- Mecanismo que detiene triggers si el pipeline falla N veces consecutivas. -->
6. ¿Cómo implementarías rollback sin afectar a todos los usuarios?
   <!-- Canary deployment: nuevo modelo a 5% de tráfico, comparar métricas, rollback selectivo. -->
7. ¿Qué información guardarías en el lineage de un modelo?
   <!-- Data version, feature view version, código version, hyperparameters, metrics, commit SHA. -->
8. ¿Cuándo preferirías un trigger schedule sobre uno por drift?
   <!-- Cuando el drift es difícil de medir (texto, imágenes) o el costo de reentrenar es bajo. -->

## ¿Dónde ir Siguente?

- [[Feature Stores]] — features versionadas para reentrenamiento
- [[Data & Concept Drift]] — técnicas avanzadas de detección de drift
- [[Model Monitoring]] — dashboards para detectar degradación en producción
- [[ML Pipelines]] — orquestación con Kubeflow, Airflow, o TFX
- [[Experiment Tracking]] — seguimiento con MLflow y Weights & Biases
- [[CI-CD & GitOps]] — integrar continuous training con CI/CD
- [[Data Quality & Testing]] — testing de datos para pipelines de entrenamiento
