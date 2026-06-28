---
tags: [time-series, anomaly-detection, monitoring]
status: growing
created: 2026-06-27
---

# Time Series Anomaly Detection

## 1. Escenario de aprendizaje

Tu equipo monitorea la latencia de una API. Normalmente es ~200ms. De repente sube a 2000ms. No es un error — la API sigue respondiendo. Pero el rendimiento es inaceptable. Necesitás que un sistema detecte automáticamente estos "picos silenciosos" y alerte antes de que los usuarios se quejen.

## 2. Tipos de anomalías temporales

| Tipo | Ejemplo | Visual |
|------|---------|--------|
| **Point** | Pico único de latencia | Un valor muy alto, el resto normal |
| **Contextual** | 30°C en Buenos Aires en julio | Normal en verano, anómalo en invierno |
| **Pattern** | Ritmo cardíaco que deja de tener variabilidad | Secuencia completa anómala |
| **Level shift** | Latencia salta de 200ms a 400ms y se queda | Media cambia permanentemente |
| **Trend change** | CPU crece 2% por día en vez de 0.5% | Pendiente cambia |

## 3. Métodos estadísticos

### 3.1 Z-score

$$z_t = \frac{x_t - \mu}{\sigma}$$

Si $|z_t| > 3$, es anomalía. Simple, pero asume distribución normal y no tolera seasonality.

```python
def zscore_anomalies(series, threshold=3):
    z = (series - series.mean()) / series.std()
    return np.where(np.abs(z) > threshold)[0]
```

**Problema**: si la serie tiene una tendencia creciente, la media se desplaza y todo parece anómalo.

### 3.2 Modified Z-score (MAD)

Usa mediana y MAD (robusto a outliers):

$$z'_t = \frac{0.6745(x_t - \text{mediana})}{\text{MAD}}$$

### 3.3 EWMA (Exponentially Weighted Moving Average)

Detecta cambios suaves en la media:

```python
def ewma_detect(series, alpha=0.3, threshold=3):
    ewma = [series.iloc[0]]
    anomalias = []
    for t in range(1, len(series)):
        val = alpha * series.iloc[t] + (1 - alpha) * ewma[-1]
        ewma.append(val)
        residuo = abs(series.iloc[t] - val)
        if residuo > threshold * series.std():
            anomalias.append(t)
    return anomalias
```

## 4. Seasonal decomposition + residual

Para series con estacionalidad: descomponé, analizá el residuo.

```python
from statsmodels.tsa.seasonal import STL

stl = STL(series, period=24, robust=True)  # datos horarios, ciclo diario
result = stl.fit()

residuo = result.resid
# Si el residuo supera k*σ, es anomalía
threshold = 3 * residuo.std()
anomalias = np.where(np.abs(residuo) > threshold)[0]

print(f"Detectadas {len(anomalias)} anomalías")
```

**Ventaja**: elimina estacionalidad y tendencia antes de analizar. Ideal para server metrics (CPU, memoria, latencia).

## 5. Autoencoder reconstruction error

Entrenás un autoencoder en datos normales. En inference, si la reconstrucción es mala, es anomalía.

```python
class AnomalyAE(nn.Module):
    def __init__(self, input_dim=60):  # ventana de 60 puntos
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Entrenar solo en datos normales
model = AnomalyAE()
for X_batch in dataloader_normal:
    loss = F.mse_loss(model(X_batch), X_batch)
    loss.backward()
    optimizer.step()

# Inference: ventanas con alta reconstruction error = anomalías
reconstruction_error = F.mse_loss(model(X_test), X_test, reduction='none').mean(dim=1)
anomalias = reconstruction_error > percentile(reconstruction_error, 95)
```

**Ventaja**: no necesita etiquetas. **Desventaja**: puede aprender a reconstruir anomalías también si no separás cuidadosamente los datos de entrenamiento.

## 6. Matrix Profile (STUMPY)

Computa la distancia de cada subsecuencia de la serie a su vecino más cercano. Anomalías = subsecuencias con vecinos lejanos.

```python
import stumpy

# m = longitud de la subsecuencia (ej: 50 puntos)
mp = stumpy.stump(series, m=50)

# mp[:, 0] = matrix profile (distancia al vecino más cercano)
anomaly_score = mp[:, 0]

# Las top-5 anomalías son los picos del profile
top_anomalias = np.argsort(anomaly_score)[-5:]
```

## 7. Threshold selection

| Método | Cómo funciona | Cuándo usarlo |
|--------|---------------|---------------|
| **Fijo** | $|residuo| > k$ | Serie estable, sin cambios de varianza |
| **Percentil** | Top 1% de error | Querés tasa de alerta fija |
| **Rolling** | $k \times \sigma_{window}$ | Varianza cambia con el tiempo |
| **POT** | Extreme Value Theory | Querés adaptación automática |

## 8. Common Mistakes

1. **No considerar estacionalidad**: lo que es normal a las 3pm puede ser anómalo a las 3am. Descomponé primero.
2. **Threshold fijo en serie no estacionaria**: la varianza cambia con el tiempo. Usá rolling window threshold.
3. **Autoencoder entrenado con anomalías incluidas**: el autoencoder aprende a reconstruir anomalías y no las detecta. Limpiá los datos de entrenamiento.
4. **False positives por level shifts**: un cambio permanente de media se marca como anomalía continua. Usá change point detection (PELT, CUSUM) para shifts.

## 9. Check Your Understanding

1. Tu serie tiene un pico a las 2pm todos los días (hora pico). ¿Z-score simple lo marcaría como anomalía? (Sí, aunque no lo sea — necesitás descomposición estacional)
2. ¿Cuándo preferirías un autoencoder a STL+residual? (Cuando la serie es multivariada o tiene relaciones complejas entre features)
3. Threshold basado en percentil vs basado en desvío estándar: ¿cuál es más robusto a outliers?

## 10. Summary

La detección de anomalías en series temporales va de métodos simples (z-score) a complejos (autoencoders). El pipeline: descomponer (STL), analizar residuo, threshold adaptativo. Para series estables, z-score o EWMA alcanzan. Para series con estacionalidad fuerte, STL + residual. Para alta dimensionalidad, autoencoder. El threshold debe adaptarse a cambios en la varianza.

## 11. Where to Go Next

- [[Data & Concept Drift]] — detección de cambios en distribuciones
- [[Model Monitoring]] — aplicar anomalías a outputs de modelos
- [[Classical Forecasting Methods]] — usar modelos de forecast para residuos
- [[Observability]] — dashboards y alertas
- [[Time Series Fundamentals]] — descomposición y estacionariedad
- [[Deep Learning for Time Series]] — autoencoders y TCN para anomalías
- [[Statistics]] — tests estadísticos para detección de outliers
