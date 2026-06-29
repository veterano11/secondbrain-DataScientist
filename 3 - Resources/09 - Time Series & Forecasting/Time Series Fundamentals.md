---
tags: [time-series, fundamentals, statistics]
status: growing
created: 2026-06-27
---

# Time Series Fundamentals

## 1. Escenario de aprendizaje

Tenés datos de ventas diarias de tu tienda online desde enero 2023 hasta hoy. Querés predecir ventas para los próximos 30 días. Pero cuando intentás usar un modelo de regresión común, las predicciones son pésimas. El problema: las observaciones NO son independientes. Lo que se vendió ayer influye en lo que se vende hoy.

Al terminar esta nota, vas a poder descomponer una serie temporal en sus componentes (tendencia, estacionalidad, residuo), determinar si es estacionaria, y prepararla para modelos de forecasting.

## 2. ¿Qué hace que un dato sea temporal?

En ML clásico, asumimos que las filas son independientes: el orden no importa. En series temporales, el orden **es** la información.

```
ML clásico:     sortear las filas no cambia el modelo
Series temp:    sortear destruye la estructura temporal
```

## 3. Componentes de una serie temporal

Toda serie temporal se descompone en tres componentes:

$$y_t = T_t + S_t + R_t \quad \text{(aditiva)}$$
$$y_t = T_t \times S_t \times R_t \quad \text{(multiplicativa)}$$

- **Tendencia** $T_t$: dirección de largo plazo (sube, baja, plana)
- **Estacionalidad** $S_t$: patrones periódicos (diario, semanal, anual)
- **Residuo** $R_t$: todo lo demás (idealmente ruido blanco)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Simular datos de ventas
np.random.seed(42)
fechas = pd.date_range('2023-01-01', periods=365, freq='D')
tendencia = np.linspace(100, 200, 365)                    # sube lentamente
estacionalidad = 30 * np.sin(2 * np.pi * np.arange(365) / 7)  # ciclo semanal
ruido = np.random.normal(0, 10, 365)
ventas = tendencia + estacionalidad + ruido

# Descomponer
result = seasonal_decompose(ventas, model='additive', period=7)

# Ver resultados
print(f"Tendencia: min={result.trend.min():.0f}, max={result.trend.max():.0f}")
print(f"Estacionalidad: amplitud ≈ {result.seasonal.max() - result.seasonal.min():.0f}")
print(f"Residuo std: {result.resid.std():.1f} (ideal: ~10, el ruido que simulamos)")
```

**Salida esperada:**
```
Tendencia: min=100, max=200
Estacionalidad: amplitud ≈ 60
Residuo std: 10.2 (ideal: ~10, el ruido que simulamos)
```

**Qué observar**: la tendencia captura la subida gradual. La estacionalidad captura el ciclo semanal. El residuo debería verse como ruido sin patrón — si el residuo muestra un patrón, significa que no capturaste toda la estructura.

## 4. Estacionariedad

Una serie es **estacionaria** si sus propiedades estadísticas (media, varianza, autocorrelación) NO cambian con el tiempo.

### 4.1 Por qué importa

La mayoría de los modelos de forecasting (ARIMA, auto-regresivos) **requieren** estacionariedad porque modelan desviaciones alrededor de una media constante. Si la media cambia con el tiempo, el modelo se desactualiza constantemente.

### 4.2 Cómo detectarla

**Augmented Dickey-Fuller (ADF) test:**

- $H_0$: la serie tiene raíz unitaria (NO es estacionaria)
- $H_1$: la serie es estacionaria
- Si $p < 0.05$, rechazamos $H_0$ → la serie es estacionaria

```python
from statsmodels.tsa.stattools import adfuller

# Simular serie no estacionaria (random walk)
random_walk = np.cumsum(np.random.normal(0, 1, 200))

result = adfuller(random_walk)
print(f"ADF statistic: {result[0]:.2f}")
print(f"p-value: {result[1]:.4f}")
if result[1] < 0.05:
    print("→ Serie estacionaria (rechazamos H0)")
else:
    print("→ Serie NO estacionaria (no podemos rechazar H0)")
```

**Salida esperada:**
```
ADF statistic: -1.23
p-value: 0.6593
→ Serie NO estacionaria (no podemos rechazar H0)
```

Un random walk típicamente NO es estacionario — su media y varianza crecen con el tiempo.

### 4.3 Cómo hacerla estacionaria

**Diferencia simple**: $y'_t = y_t - y_{t-1}$

```python
# Primera diferencia
diff_1 = np.diff(ventas, n=1)  # ahora tenemos 364 valores

# Verificar si es estacionaria ahora
result_diff = adfuller(diff_1)
print(f"Diferencia - p-value: {result_diff[1]:.4f}")
```

**Otras técnicas**:
- **Log transform**: estabiliza varianza cuando la magnitud crece con el nivel
- **Diferenciación estacional**: $y'_t = y_t - y_{t-s}$ para eliminar estacionalidad
- **Detrending**: restar la tendencia ajustada (regresión lineal, LOESS)

```python
# Diferenciación estacional (periodo semanal = 7)
diff_s = ventas[7:] - ventas[:-7]
```

## 5. Autocorrelación

La autocorrelación mide cómo se relaciona un valor con sus valores pasados.

### 5.1 ACF (Autocorrelation Function)

Correlación entre $y_t$ y $y_{t-k}$ para diferentes $k$.

**Interpretación**: si ACF tiene un pico en lag=7 con datos diarios, hay un patrón semanal.

### 5.2 PACF (Partial ACF)

Correlación entre $y_t$ y $y_{t-k}$ eliminando el efecto de los lags intermedios.

**Útil para identificar orden AR**:

| Patrón ACF | Patrón PACF | Modelo sugerido |
|-----------|-------------|-----------------|
| Decae gradualmente | Corta en lag p | AR(p) |
| Corta en lag q | Decae gradualmente | MA(q) |
| Decae gradualmente | Decae gradualmente | ARMA(p,q) |

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ventas, lags=40, ax=axes[0])
plot_pacf(ventas, lags=40, ax=axes[1])
plt.show()
```

## 6. Ruido blanco

El residuo ideal de un modelo de series temporales es **ruido blanco**:
- Media 0
- Varianza constante
- Sin autocorrelación significativa

Si los residuos de tu modelo NO son ruido blanco (tienen patrón), tu modelo dejó información sin capturar.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Test de Ljung-Box para ruido blanco
# H0: los residuos son independientes (ruido blanco)
resultado = acorr_ljungbox(residuos, lags=[10])
print(f"p-value = {resultado['lb_pvalue'].values[0]:.3f}")
# Si p > 0.05, no podemos rechazar H0 → los residuos son ruido blanco
```

## 7. Pipeline completo

```python
def preparar_serie(ventas, periodo=7):
    """Pipeline para preparar serie temporal para modelado."""

    # 1. Graficar para inspección visual
    plt.plot(ventas); plt.title("Datos crudos")

    # 2. Descomponer para entender componentes
    result = seasonal_decompose(ventas, model='additive', period=periodo)

    # 3. Test de estacionariedad
    p_valor = adfuller(ventas)[1]
    print(f"Estacionaria? p={p_valor:.4f}")

    # 4. Si no es estacionaria, diferenciar
    if p_valor >= 0.05:
        ventas = np.diff(ventas, n=1)
        print("Aplicada diferencia de orden 1")

    return ventas
```

## 8. Common Mistakes

1. **Leaky validation**: dividir datos temporales con train_test_split aleatorio es trampa. El modelo ve futuro para predecir pasado. Usá TimeSeriesSplit.

2. **No graficar antes de modelar**: si no visualizás la serie, no sabés si hay estacionalidad, outliers, o tendencia que cambia. Siempre graficar primero.

3. **Confundir correlación con causalidad**: dos series pueden tener alta correlación sin relación causal (por ejemplo, ventas de helado y ahogados en piletas — ambos suben en verano). Diferenciar ayuda a evitar esto.

4. **Over-differencing**: diferenciar una serie que ya era estacionaria introduce autocorrelación artificial y empeora el forecast.

5. **Asumir periodicidad incorrecta**: datos diarios con ciclo semanal → periodo=7. Datos mensuales con ciclo anual → periodo=12. Datos horarios con ciclo diario → periodo=24.

## 9. Check Your Understanding

1. Graficás la ACF de tus residuos y ves picos significativos en lags 1, 2, y 3. ¿Qué te dice esto sobre tu modelo?
2. Generás una serie con $y_t = y_{t-1} + \varepsilon_t$ (random walk). ¿Es estacionaria? ¿Por qué?
3. Tenés datos de temperatura diaria por 3 años. ¿Qué periodicidad esperás y cómo la verificás?
4. Aplicás diferencia simple a una serie estacionaria. ¿Qué efecto tiene?

**Respuestas rápidas:**
1. El modelo dejó información temporal sin capturar — necesita más lags AR o MA.
2. No. La varianza crece con el tiempo. Diferenciar la hace estacionaria.
3. Periodo = 365 (anual). Se ve en ACF con pico en lag 365.
4. Introduce autocorrelación negativa artificial → empeora el modelo.

## 10. Resumen

Las series temporales tienen estructura temporal que los modelos estándar ignoran. Las tres componentes (tendencia, estacionalidad, residuo) se separan con descomposición. La estacionariedad es un requisito para la mayoría de los modelos de forecasting — se logra con diferenciación. La ACF y PACF guían la selección del modelo. El pipeline correcto: graficar → descomponer → testear estacionariedad → diferenciar si es necesario → modelar.

## 11. Where to Go Next

- [[Classical Forecasting Methods]] — ARIMA, Prophet, Holt-Winters
- [[Deep Learning for Time Series]] — LSTM, TCN, N-BEATS
- [[Time Series Anomaly Detection]] — detectar outliers en datos temporales
- [[Data & Concept Drift]] — series temporales monitorean drift en producción
- [[Experiment Tracking]] — cómo comparar modelos de forecasting
- [[Statistics]] — fundamentos estadísticos de series temporales
- [[Feature Engineering]] — creación de lags y rolling features
