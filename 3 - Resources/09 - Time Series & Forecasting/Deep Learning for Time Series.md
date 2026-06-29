---
tags: [time-series, deep-learning, lstm, tcn]
status: growing
created: 2026-06-27
---

# Deep Learning for Time Series

## 1. Escenario de aprendizaje

Trabajás en forecasting de demanda energética para una red eléctrica. Tenés 500 series simultáneas (consumo por ciudad), cada una con patrones complejos, interacciones no lineales, y dependencias de largo plazo. ARIMA no escala a 500 series. Prophet no captura las no linealidades. Necesitás deep learning.

## 2. De serie temporal a supervisado

El primer paso es convertir la serie a ventanas (sliding windows):

```python
def crear_ventanas(series, window=30, horizon=7):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window])
        y.append(series[i+window:i+window+horizon])
    return np.array(X), np.array(y)

# Ejemplo: con 365 días, window=30, horizon=7
# → (365-30-7+1) = 329 muestras de entrenamiento
```

**Arquitectura típica**: la salida tiene horizonte de 7 puntos, no solo 1. Esto se llama **multi-step forecasting**.

## 3. LSTM

La LSTM (Long Short-Term Memory) fue diseñada para evitar el vanishing gradient de las RNNs vanilla.

### 3.1 Componentes internos

```python
class LSTMCell:
    """Una celda LSTM en pseudocódigo"""
    def forward(self, x_t, h_prev, c_prev):
        # Forget gate: qué olvidar del estado anterior
        f = sigmoid(W_f @ [h_prev, x_t] + b_f)
        # Input gate: qué recordar de la nueva entrada
        i = sigmoid(W_i @ [h_prev, x_t] + b_i)
        # Candidate: información nueva propuesta
        c_tilde = tanh(W_c @ [h_prev, x_t] + b_c)
        # Nuevo estado celda
        c = f * c_prev + i * c_tilde
        # Output gate: qué output dar
        o = sigmoid(W_o @ [h_prev, x_t] + b_o)
        h = o * tanh(c)
        return h, c
```

### 3.2 Uso práctico

```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden=64, num_layers=2, horizon=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.lstm(x)           # (B, T, hidden)
        last = out[:, -1, :]            # (B, hidden)
        return self.fc(last)            # (B, horizon)
```

**Limitaciones**: no paralelizable (procesa secuencial), difícil de entrenar para secuencias > 500 pasos.

## 4. TCN (Temporal Convolutional Network)

Convoluciones causales + dilataciones para lograr gran receptive field sin recurrencia.

```python
class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel,
                              padding=(kernel-1)*dilation,
                              dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_c)

    def forward(self, x):
        # Causal: padding solo al inicio, no al final
        out = self.conv(x)
        out = out[:, :, :- (self.conv.kernel_size[0] - 1)]  # causal
        return self.norm(self.relu(out))
```

**Receptive field**: $1 + \sum_{i=0}^{L-1} (k-1) \cdot d^i$

Con $k=3, d=2, L=8$: $1 + 2 \times (1+2+4+8+16+32+64+128) = 511$ pasos. Enorme receptive field con solo 8 capas.

**Ventaja sobre LSTM**: paralelizable (convoluciones), gradientes estables, receptive field configurable.

## 5. N-BEATS

Arquitectura basada puramente en MLPs (sin recurrencia ni convolución). Cada stack produce un "backcast" (lo que elimina de la entrada) y un "forecast" (lo que predice). Stacks apilados modelan residuos.

```text
Input → Stack 1: backcast(1) + forecast(1)
              ↓ (restar backcast)
       Stack 2: backcast(residuo) + forecast(2)
              ↓ (restar backcast)
       Stack 3: ...
              ↓
       Suma de forecasts = predicción final
```

**Interpretabilidad**: si los stacks usan bases polinomiales (tendencia) y Fourier (estacionalidad), podés ver qué parte de la predicción viene de cada componente.

## 6. Temporal Fusion Transformer (TFT)

Estado del arte para forecasting multi-horizon con múltiples series. Combina:
- Selección de variables: aprende qué features importan
- LSTM encoder-decoder: procesa pasado y futuro conocido
- Multi-head attention: captura dependencias de largo plazo
- Quantile outputs: da 10°, 50°, 90° percentiles (incertidumbre)

## 7. DeepAR (Amazon)

Modelo probabilístico: cada paso predice los parámetros de una distribución (ej: Normal(μ, σ)), no un valor puntual.

$\hat{y}_t \sim \mathcal{N}(\mu_t, \sigma_t)$

Útil cuando la incertidumbre es importante (inventarios, finanzas).

## 8. Common Mistakes

1. **No escalar por serie**: si tenés 500 series con escalas distintas, estandarizá cada una por separado (z-score por serie).
2. **Usar future data como feature**: features como "temperatura del día siguiente" solo sirven si las tenés en tiempo real.
3. **Overfitting en ventanas**: más ventanas de entrenamiento ≠ más información independiente. Las ventanas solapadas están correlacionadas.
4. **Validación temporal**: usá expanding window, no KFold aleatorio.

## 9. Check Your Understanding

1. TCN con d=2 tiene receptive field exponencial. ¿Cuántas capas se necesitan para RF=1024 con k=3? (L=10: 1+2×(1+2+...+512) = 2047)
2. ¿Por qué LSTM no es paralelizable y TCN sí? (LSTM procesa un paso a la vez; TCN usa convoluciones que se computan en paralelo)
3. ¿Cuándo usarías N-BEATS en vez de TFT? (Cuando querés interpretabilidad y no necesitas features exógenas)

## 10. Resumen

Deep learning para time series transforma forecasting secuencial en supervisado con ventanas. LSTM fue el primer éxito pero es secuencial y caro. TCN ofrece paralelismo y receptive field grande. N-BEATS es interpretable (solo MLPs). TFT es el estado del arte para forecasting complejo con múltiples series y features. La regla de oro: para pronósticos simples (<100 puntos), usá clásicos. Para patrones complejos y muchas series, usá DL.

## 11. Where to Go Next

- [[Classical Forecasting Methods]] — baseline para comparar
- [[Time Series Fundamentals]] — ventanas, estacionariedad
- [[RNNs & Sequence Models]] — la base teórica de LSTM
- [[Transformers]] — atención que TFT adapta a tiempo
- [[Training Techniques]] — entrenamiento de redes recurrentes y convolucionales
- [[Model Evaluation]] — evaluación de forecasting multi-step
- [[Feature Engineering]] — ventanas y features temporales
