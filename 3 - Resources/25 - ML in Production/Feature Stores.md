---
tags:
  - ml-engineering
  - mlops
  - feature-stores
  - production-ml
status: seedling
created: 2026-06-28
---

# Feature Stores

## 1. Escenario de aprendizaje

Trabajas en el equipo de fraude de una fintech. Tu modelo detecta transacciones fraudulentas en tiempo real — cada compra debe evaluarse en <200ms. Durante el entrenamiento usaste DataFrames de Spark con 200 features calculadas sobre lotes históricos. En producción, las transacciones llegan una por una desde Kafka, y necesitas exactamente las mismas features.

El problema clásico: las features de entrenamiento se calcularon con `pd.DataFrame.groupby()`, joins y ventanas de tiempo retrospectivas. En producción no tienes el lote completo — tienes un solo evento. Recalcular cada feature en el endpoint de inferencia es frágil y lleva a inconsistencias entre train y serve.

Un feature store unifica features offline (entrenamiento) y online (inferencia), garantizando que el cálculo sea idéntico en ambos entornos.

## 2. Requisitos

- Python 3.10+, Feast (`pip install feast`), Redis
- Conceptos básicos de MLOps y pipelines de datos

## 3. Online vs Offline Features

### Offline Features

Features calculadas en lotes (diarios, horarios) sobre data warehouses (BigQuery, Snowflake) o data lakes (S3, GCS). Se usan para entrenar modelos, backtesting, y batch inference.

```python
offline_features = pd.DataFrame({
    "user_id": [1, 2, 3],
    "amount_avg_7d": [150.0, 3200.0, 45.50],
    "tx_count_1h": [3, 1, 12],
    "event_timestamp": pd.to_datetime(["2026-06-27 10:00", "2026-06-27 10:05", "2026-06-27 10:10"])
})
print(offline_features)
```

**Salida esperada:**
```
   user_id  amount_avg_7d  tx_count_1h  event_timestamp
0        1         150.0            3 2026-06-27 10:00:00
1        2        3200.0            1 2026-06-27 10:05:00
2        3          45.5           12 2026-06-27 10:10:00
```

### Online Features

Se calculan en tiempo real y se sirven con latencia de milisegundos desde Redis, DynamoDB o Firestore. Se usan para inferencia en APIs síncronas y scoring de eventos streaming.

```python
online_features = {"user:1": {"amount_avg_7d": 150.0, "tx_count_1h": 3}}
print(online_features["user:1"])
```

**Salida esperada:** `{'amount_avg_7d': 150.0, 'tx_count_1h': 3}`

### Latencia y Consistencia

| Aspecto | Offline | Online |
|---------|---------|--------|
| Latencia | Minutos a horas | < 50ms |
| Volumen | TBs por lote | KBs por request |
| Actualización | Batch (1h-24h) | Streaming / CDC |
| Almacenamiento | Parquet, ORC, Avro | Redis, DynamoDB |

**Regla de oro**: la feature offline y online deben dar el mismo valor para la misma entidad en el mismo instante. Si difieren, el modelo entrena con una distribución distinta a la que ve en producción.

## 4. Point-in-Time Correct Joins

El problema más difícil en feature stores. Si entrenas con un `JOIN` directo entre transacciones y features calculadas diariamente, terminas con data leakage — el modelo aprende del futuro.

```python
transactions = pd.DataFrame({
    "user_id": [1, 1, 1], "amount": [100, 200, 150],
    "tx_timestamp": pd.to_datetime(["2026-06-27 10:00", "2026-06-27 14:00", "2026-06-28 09:00"])
})
daily_features = pd.DataFrame({
    "user_id": [1, 1, 1], "amount_avg_7d": [120.0, 130.0, 140.0],
    "feature_timestamp": pd.to_datetime(["2026-06-27 00:00", "2026-06-28 00:00", "2026-06-29 00:00"])
})

bad_join = transactions.merge(daily_features, on="user_id")
print(bad_join)
# 6 filas — cada transacción aparece con features pasadas Y futuras
```

**Salida esperada:**
```
   user_id  amount        tx_timestamp  amount_avg_7d feature_timestamp
0        1     100 2026-06-27 10:00:00         120.0        2026-06-27
1        1     100 2026-06-27 10:00:00         130.0        2026-06-28  ← fuga
2        1     100 2026-06-27 10:00:00         140.0        2026-06-29  ← fuga
```

### Point-in-Time Correct Join

Para cada transacción, tomar la feature más reciente ANTERIOR a la transacción.

```python
transactions = transactions.sort_values("tx_timestamp")
daily_features = daily_features.sort_values("feature_timestamp")

correct = []
for _, tx in transactions.iterrows():
    mask = (daily_features["user_id"] == tx["user_id"]) & \
           (daily_features["feature_timestamp"] <= tx["tx_timestamp"])
    valid = daily_features[mask]
    if not valid.empty:
        row = tx.to_dict()
        row["amount_avg_7d"] = valid.iloc[-1]["amount_avg_7d"]
        correct.append(row)

print(pd.DataFrame(correct))
```

**Salida esperada:**
```
   user_id  amount        tx_timestamp  amount_avg_7d
0        1     100 2026-06-27 10:00:00         120.0
1        1     200 2026-06-27 14:00:00         120.0
2        1     150 2026-06-28 09:00:00         130.0
```

Feast, Tecton y otros feature stores implementan esto automáticamente.

## 5. Feast: Feature Repository

Feast es un feature store open-source que define features como código y provee serving offline/online.

```bash
pip install feast
feast init my_feature_repo
cd my_feature_repo
```

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32

user = Entity(name="user_id", join_keys=["user_id"])

user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="amount_avg_7d", dtype=Float32),
        Field(name="tx_count_1h", dtype=Int32),
    ],
    source=FileSource(path="data/transaction_stats.parquet", timestamp_field="event_timestamp"),
    online=True,
)
```

### Serving Offline

```python
store = FeatureStore(repo_path=".")
training_df = store.get_historical_features(
    entity_df=pd.DataFrame({
        "user_id": [1, 2],
        "event_timestamp": pd.to_datetime(["2026-06-27 10:00", "2026-06-27 10:05"])
    }),
    features=["user_transaction_features:amount_avg_7d"]
).to_df()
print(training_df)
```

**Salida esperada:**
```
   user_id         event_timestamp  amount_avg_7d
0        1 2026-06-27 10:00:00          150.0
1        2 2026-06-27 10:05:00         3200.0
```

### Serving Online (Redis)

```python
store.materialize_incremental(end_date=datetime(2026, 6, 28, 12, 0, 0))

features = store.get_online_features(
    features=["user_transaction_features:amount_avg_7d"],
    entity_rows=[{"user_id": 1}]
).to_dict()
print(features)
```

**Salida esperada:** `{'user_id': [1], 'amount_avg_7d': [150.0]}`

## 6. Feature Serving: Arquitectura

```python
class FeatureStoreService:
    def __init__(self):
        self.online_store = {}
        self.offline_store = {}

    def get_features(self, entity_id: str, feature_names: list[str], is_online: bool = True) -> dict:
        store = self.online_store if is_online else self.offline_store
        data = store.get(entity_id, {})
        return {f: data.get(f) for f in feature_names}

svc = FeatureStoreService()
svc.online_store["user:1"] = {"amount_avg_7d": 150.0}
print(svc.get_features("user:1", ["amount_avg_7d"]))
```

**Salida esperada:** `{'amount_avg_7d': 150.0}`

### API Unificada

```bash
GET /features/v1/user_id=1?features=amount_avg_7d,tx_count_1h
```
```json
{"user_id": 1, "features": {"amount_avg_7d": 150.0, "tx_count_1h": 3}}
```

## 7. Feature Validation

Las features pueden degradarse silenciosamente. Validación de schema y detección de drift previenen esto.

```python
from scipy.stats import ks_2samp
import numpy as np

train_amounts = np.random.lognormal(mean=5.0, sigma=1.0, size=1000)
prod_amounts = np.random.lognormal(mean=6.0, sigma=1.5, size=1000)

stat, p_value = ks_2samp(train_amounts, prod_amounts)
print(f"KS={stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("Drift detectado en amount_avg_7d")
```

**Salida esperada:** `KS=0.342, p=0.0000 \n Drift detectado en amount_avg_7d`

```python
class FeatureMonitor:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.reference = {}

    def register(self, feature: str, values: np.ndarray):
        self.reference[feature] = values

    def check_drift(self, feature: str, current: np.ndarray) -> bool:
        stat, p = ks_2samp(self.reference[feature], current)
        if p < self.threshold:
            print(f"[ALERT] {feature} drifted (p={p:.4f})")
            return True
        return False

monitor = FeatureMonitor()
monitor.register("amount_avg_7d", train_amounts)
monitor.check_drift("amount_avg_7d", prod_amounts)
```

**Salida esperada:** `[ALERT] amount_avg_7d drifted (p=0.0000)`

## 8. Common Mistakes

### Mistake 1: Point-in-Time Join Incorrecto — JOIN directo sin timestamp produce data leakage.
**Solución**: Usar `asof_join` en pandas o la funcionalidad point-in-time del feature store.

### Mistake 2: Features Diferentes entre Train y Serve — calcular features de una forma en el notebook y de otra en producción.
**Solución**: Definir features en un feature store que genere código idéntico para ambos entornos.

### Mistake 3: Ignorar TTL en Online Store — features caducan y el store devuelve `None` o valores stale.
**Solución**: Configurar `ttl` en FeatureViews. Monitorear frescura de features.

### Mistake 4: No Validar Features en Producción — valores nulos, infinitos, o distribuciones cambiadas pasan desapercibidos.
**Solución**: Great Expectations + alertas automáticas + monitoreo de drift.

## Resumen

- Un **feature store** unifica features offline y online, garantizando consistencia train/serve.
- **Point-in-time correct joins** evitan data leakage respetando la línea temporal.
- **Feast** define features como código y ofrece serving offline (BigQuery, S3) y online (Redis, DynamoDB).
- La **validación de features** con Great Expectations y KS test previene degradaciones silenciosas.

## Comprueba tu Conocimiento

1. ¿Cuál es la diferencia principal entre online features y offline features?
   <!-- Online: baja latencia, Redis/DynamoDB. Offline: grandes volúmenes, batch, data warehouse. -->
2. ¿Qué problema resuelven los point-in-time correct joins?
   <!-- Evitan data leakage al asegurar que solo se usen features calculadas antes del timestamp del evento. -->
3. ¿Cómo garantiza Feast que las features de entrenamiento y producción sean idénticas?
   <!-- Define features como código que se ejecuta igual en offline y online. -->
4. ¿Qué métrica usarías para detectar drift en una feature numérica continua?
   <!-- KS test (Kolmogorov-Smirnov) o Population Stability Index (PSI). -->
5. ¿Por qué es importante el TTL en un feature store online?
   <!-- Para evitar servir features stale; si no se actualizan dentro del TTL, se marcan como no disponibles. -->
6. ¿Qué rol cumple Redis en un feature store?
   <!-- Almacén de features online con lecturas en microsegundos para inferencia en tiempo real. -->
7. ¿Cuándo usarías batch inference en lugar de online inference?
   <!-- Cuando la latencia no es crítica (recomendaciones diarias, scoring masivo) y el volumen es grande. -->
8. ¿Cómo detectarías que una feature dejó de actualizarse en el online store?
   <!-- Monitorear el timestamp de última actualización y alertar si supera un umbral configurable. -->

## ¿Dónde ir Siguente?

- [[Feature Engineering]] — técnicas de creación de features
- [[Data Pipelines & ETL]] — orquestación del cálculo de features offline
- [[Model Serving]] — consumo de features online desde un endpoint de inferencia
- [[Data Quality & Testing]] — validación de datos y features con Great Expectations
- [[Data & Concept Drift]] — monitoreo continuo de features y predicciones
- [[Streaming & Event-Driven (Kafka)]] — features en tiempo real con Kafka Streams
- [[Deploy Strategies for ML]] — canary releases y shadow deployments de features nuevas
