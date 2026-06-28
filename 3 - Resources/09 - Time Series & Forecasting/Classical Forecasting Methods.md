---
tags: [time-series, forecasting, arima, prophet]
status: growing
created: 2026-06-27
---

# Classical Forecasting Methods

## 1. Escenario de aprendizaje

Tenés 3 años de ventas diarias de tu tienda. Querés pronosticar los próximos 30 días. Podrías usar una LSTM, pero con ~1000 datos, un modelo clásico (ARIMA, Prophet) probablemente funcione mejor, sea más rápido de entrenar, y te dé intervalos de confianza interpretables.

Los modelos clásicos son la primera línea de ataque para forecasting.

## 2. Baseline: Moving Average

El forecast más simple posible. No es bueno, pero es el piso contra el que medís todo lo demás.

**Simple**: promedio de los últimos $n$ valores. **Exponencial**: da más peso a observaciones recientes.

```python
def naive_forecast(series, n=7):
    return series[-n:].mean()

# MASE = MAE(modelo) / MAE(naive)
# Si MASE < 1, tu modelo supera al baseline
```

## 3. Exponential Smoothing (Holt-Winters)

### 3.1 Single (solo nivel)

$$\ell_t = \alpha y_t + (1-\alpha) \ell_{t-1}$$

Suaviza la serie sin necesidad de ventana. $\alpha$ cerca de 1 → se adapta rápido. Cerca de 0 → muy suave.

### 3.2 Double (nivel + tendencia)

Agrega un componente de tendencia $b_t$:

$$\ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})$$
$$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1}$$

### 3.3 Triple / Holt-Winters (nivel + tendencia + estacionalidad)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    ventas,
    trend='add',
    seasonal='add',
    seasonal_periods=7,   # datos diarios, ciclo semanal
).fit()

forecast = model.forecast(steps=30)
print(f"Pronóstico próximo mes: {forecast.mean():.0f} ± {forecast.std():.0f}")
```

**Cuándo usarlo**: datos con estacionalidad clara y sin cambios abruptos. No necesita tuning.

## 4. ARIMA(p,d,q)

El modelo más usado de la estadística clásica.

### 4.1 Componentes

| Componente | Significado | Identificación |
|------------|-------------|----------------|
| **AR(p)** | Autoregresivo: $y_t$ depende de $y_{t-1}, ..., y_{t-p}$ | PACF corta en lag p |
| **I(d)** | Diferenciación: $d$ diferencias para estacionariedad | ADF test |
| **MA(q)** | Media móvil: $y_t$ depende de errores pasados $\varepsilon_{t-1}, ..., \varepsilon_{t-q}$ | ACF corta en lag q |

### 4.2 Cómo elegir p, d, q

```python
from pmdarima import auto_arima

# 1. Dejá que auto_arima busque
model = auto_arima(ventas, seasonal=True, m=7, trace=True)
print(model.summary())

# 2. También podés usar ACF/PACF manualmente
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ventas, lags=40)
plot_pacf(ventas, lags=40)
# ACF decae, PACF corta en 2 → AR(2)
```

### 4.3 Paso a paso

```python
from statsmodels.tsa.arima.model import ARIMA

# Asumiendo d=1 (diferenciación simple)
model = ARIMA(ventas, order=(2, 1, 1)).fit()

# Diagnóstico: residuos deben ser ruido blanco
residuos = model.resid
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(residuos, lags=[10]))
# p > 0.05 → residuos son ruido blanco → modelo ok

# Forecast
forecast = model.forecast(steps=30)
```

### 4.4 SARIMA

Agrega componentes estacionales: ARIMA$(p,d,q)(P,D,Q)_m$

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(ventas, order=(1,1,1), seasonal_order=(1,1,1,7)).fit()
# Seasonal_order: (P, D, Q, m=periodo)
```

## 5. Prophet (Meta)

Diseñado para forecasting business con:
- Múltiples estacionalidades (anual, semanal, diaria)
- Efectos de feriados
- Changepoints en la tendencia

$$y_t = g(t) + s(t) + h(t) + \varepsilon_t$$

```python
from prophet import Prophet

df = pd.DataFrame({'ds': fechas, 'y': ventas.values})

model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_country_holidays('US')  # feriados
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
# Prophet grafica datos históricos, forecast, e intervalos de confianza
```

**¿Cuándo Prophet es mejor que ARIMA?**

| Situación | Ganador |
|-----------|---------|
| Estacionalidad múltiple (diaria + semanal + anual) | Prophet |
| Efectos de feriados | Prophet |
| Changepoints en tendencia | Prophet |
| Series largas (>2 años) con estacionalidad simple | ARIMA |
| Series cortas (<100 puntos) | Holt-Winters |

## 6. Evaluación

| Métrica | Fórmula | Nota |
|---------|---------|------|
| MAE | $|y_t - \hat{y}_t|$ | Promedio del error absoluto |
| RMSE | $\sqrt{(y_t - \hat{y}_t)^2}$ | Penaliza errores grandes |
| MAPE | $\frac{|y_t - \hat{y}_t|}{y_t}$ | Error porcentual; falla si $y_t=0$ |
| MASE | $\frac{\text{MAE(modelo)}}{\text{MAE(naive)}}$ | < 1 → modelo mejor que baseline |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAE={mae:.0f}, RMSE={rmse:.0f}, MAPE={mape:.1f}%")
```

## 7. Common Mistakes

1. **MAPE con ceros**: si algún valor real es 0, MAPE es infinito. Usá MASE o sMAPE.
2. **No revisar residuos**: un ARIMA puede dar forecast aunque los residuos tengan autocorrelación. Si los residuos no son ruido blanco, el modelo está mal especificado.
3. **Validación temporal incorrecta**: TimeSeriesSplit (no KFold aleatorio). Los folds respetan el orden temporal.
4. **Prophet con changepoints mal calibrados**: `changepoint_prior_scale` alto (=0.05) → sobreajusta la tendencia. Bajo (=0.001) → no captura cambios.

## 8. Check Your Understanding

1. Si la PACF de tu serie corta abruptamente en lag 2 y la ACF decae, ¿qué modelo probás? (AR(2))
2. ¿Por qué Prophet necesita menos tuning que ARIMA? (Maneja automáticamente estacionalidad múltiple, changepoints, feriados)
3. MASE=0.7. ¿Qué significa?
4. Ajustás un ARIMA(1,1,1) y los residuos NO son ruido blanco. ¿Qué hacés?

## 9. Summary

Los modelos clásicos de forecasting son interpretables, rápidos, y requieren pocos datos. Holt-Winters captura tendencia y estacionalidad simple. ARIMA/SARIMA es el estándar para series estacionarias. Prophet maneja estacionalidades múltiples y changepoints. El flujo: baseline (naive) → modelo clásico → diagnosticar residuos → refinar. La validación temporal es obligatoria.

## 10. Where to Go Next

- [[Time Series Fundamentals]] — descomposición, estacionariedad, ACF/PACF
- [[Deep Learning for Time Series]] — LSTM, TCN cuando los clásicos se quedan cortos
- [[Time Series Anomaly Detection]] — cómo los residuos del modelo detectan anomalías
- [[Experiment Tracking]] — comparar modelos sistemáticamente
- [[Model Evaluation]] — métricas de error y validación temporal
- [[Statistics]] — teoría estadística detrás de ARIMA
- [[Feature Engineering]] — variables exógenas y calendar features
