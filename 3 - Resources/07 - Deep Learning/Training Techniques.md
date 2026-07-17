---
tags: [deep-learning, training, advanced, entrenamiento]
status: growing
created: 2026-06-27
---

# Técnicas de Entrenamiento

## 1. Escenario de aprendizaje

Has diseñado una red neuronal para clasificar imágenes de retina y diagnosticar retinopatía diabética. Conoces la arquitectura (capas convolucionales, fully connected, softmax), pero cuando comienzas a entrenar, la pérdida no disminuye o explota a NaN. Aquí es donde las técnicas de entrenamiento marcan la diferencia. Un optimizador bien elegido, una planificación de learning rate y una estrategia de normalización pueden significar la diferencia entre un modelo que converge en horas y uno que diverge a NaN.

---

## 2. Optimizadores

### 2.1 Descenso por Gradiente Estocástico (SGD)

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

```
    Pérdida
    │
    │    •
    │      •  ← descenso por gradiente
    │        •
    │          •
    │            •
    │              •  ← óptimo
    └──────────────── Pesos
```

El optimizador más simple. Cada paso se mueve directamente opuesto al gradiente.

**Ventajas**: simple, bien entendido, buena generalización.
**Desventajas**: convergencia lenta, sensible a la tasa de aprendizaje, puede quedar atrapado en puntos de silla.

### 2.2 SGD + Momentum

$$v_{t+1} = \beta v_t + \nabla L(w_t)$$
$$w_{t+1} = w_t - \eta v_{t+1}$$

Acumula gradientes pasados para construir velocidad. Si la dirección del gradiente es consistente, el momentum acelera. Si oscila, el momentum lo suaviza.

**Intuición**: una bola rodando cuesta abajo — construye momentum en direcciones consistentes y es menos afectada por ruido local.

### 2.3 Adam

$$\text{Estimación Adaptativa de Momentos}$$

El optimizador más ampliamente usado. Mantiene:
- Primer momento (media) de gradientes pasados → momentum
- Segundo momento (varianza) de gradientes pasados → tasa de aprendizaje por parámetro

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Por qué es el predeterminado**:
- Tasas de aprendizaje adaptativas por parámetro (funciona bien sin ajuste)
- Momentum incorporado
- Maneja bien gradientes dispersos
- Robusto a diferentes escalas de problemas

### 2.4 AdamW

Adam con **decaimiento de peso desacoplado**. En Adam estándar, la regularización L2 interactúa con la tasa de aprendizaje adaptativa. AdamW aplica el decaimiento de peso separadamente, lo cual es teóricamente más limpio y funciona mejor para [[Transformers]].

### 2.5 Comparación de Optimizadores

Ver [[Ajuste de Hiperparámetros]] para orientación sobre cómo establecer estos valores.

| Optimizador | Mejor para | Hiperparámetros Clave |
|---|---|---|
| **SGD** | Clasificación de imágenes, tareas simples | LR, momentum |
| **Adam** | Propósito general (predeterminado) | LR (3e-4 por defecto), β₁, β₂ |
| **AdamW** | Transformers, LLMs | LR, decaimiento de peso |
| **Lion** | Entrenamiento con memoria limitada | LR |

---

## 3. Planificación de Tasa de Aprendizaje

### 3.1 Decaimiento Coseno

$$\eta_t = \frac{1}{2} \eta_0 \left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

```
    LR
    │
 η₀ ┤•
    │  •
    │    •
    │      •
    │        •
    │          •
    │            •
  0 ┤──────────────•──→ pasos
    0              T
```

Comienza en $\eta_0$, disminuye suavemente hasta 0 en el paso $T$. Anima al modelo a explorar ampliamente al principio y a refinar al final.

### 3.2 Calentamiento Lineal + Decaimiento Coseno

Calentamiento: aumentar linealmente la LR de 0 a $\eta_0$ en los primeros $W$ pasos.
Decaimiento: decaimiento coseno de $\eta_0$ a 0 en los pasos restantes.

El calentamiento previene la inestabilidad temprana (actualizaciones grandes cuando el modelo se inicializa aleatoriamente).

### 3.3 Reducción en Meseta

Reducir la LR por un factor (ej. 0.5) cuando la pérdida de validación deja de mejorar por $P$ épocas. Simple y efectivo.

### 3.4 Un Ciclo

Calentamiento a LR alta → decaimiento coseno a LR muy baja. Enfriar a LR muy baja al final. Conocido por convergencia rápida.

---

## 4. Normalización

### 4.1 Normalización por Lotes (Batch Normalization)

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Normaliza cada característica a través del lote. Reduce el desplazamiento interno de covariable. Permite tasas de aprendizaje más altas. Agrega ligera regularización (ruido de estadísticas de lote).

**Limitación**: el tamaño del lote debe ser lo suficientemente grande para estadísticas confiables. Inestable con tamaño de lote 1 o secuencias de longitud variable.

### 4.2 Normalización por Capa (Layer Normalization)

Normaliza a través de características para cada muestra independientemente:

$$\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

Independiente del tamaño del lote. Funciona para RNNs y Transformers (donde batch norm falla por longitudes variables).

### 4.3 RMSNorm

$$y = \frac{x}{\sqrt{\text{RMS}(x) + \epsilon}} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$$

Simplificación de LayerNorm — sin centrado en la media. Más rápido, usado en Llama y LLMs modernos.

---

## 5. Técnicas de Gradiente y Regularización

Ver [[Optimización Basada en Gradiente]] para conceptos fundamentales.

| Técnica | Qué Hace | Cuándo Usar |
|---|---|---|
| **Recorte de Gradiente** | Limita la norma del gradiente para prevenir explosión | RNNs, transformers profundos, entrenamiento inestable |
| **Acumulación de Gradiente** | Suma gradientes sobre múltiples lotes | Simular lote grande con memoria GPU limitada |
| **Precisión Mixta** | FP16/BF16 con pesos maestros FP32 | Entrenamiento 2× más rápido, mitad de memoria GPU |
| **Suavizado de Etiquetas** | Suavizar etiquetas objetivo: $y' = (1-\epsilon)y + \epsilon/K$ | Clasificación, reduce sobreconfianza |
| **Profundidad Estocástica** | Eliminar capas aleatoriamente durante entrenamiento | Redes muy profundas (1000+ capas) |

---

## 6. Aprendizaje por Transferencia

Entrenar desde cero rara vez es óptimo. En cambio, empezar desde un modelo pre-entrenado:

```
    Modelo Pre-entrenado (ej: BERT)
              │
    ┌─────────┴─────────┐
    │                   │
  Congelar           Fine-tuning
  backbone           todos los parámetros
    │                   │
  Agregar            Entrenar con
  clasificador       datos nuevos
    │                   │
  Datos pocos        Datos medianos
```

| Estrategia | Qué Actualizas | Datos Necesarios | Ejemplo |
|---|---|---|---|
| **Extracción de características** | Congelar backbone, entrenar clasificador | Pocos | ResNet en dataset de imágenes personalizado |
| **Fine-tuning completo** | Todos los parámetros | Medianos | BERT para análisis de sentimiento |
| **LoRA** | Adaptadores de bajo rango | Pocos | Llama para tarea de chat personalizada |
| **Adapter** | Capas pequeñas insertadas | Pocos | Cualquier modelo grande |

---

## 7. Entrenamiento Distribuido

Para modelos demasiado grandes para una GPU:

```
    Modelo Grande
         │
    ┌────┼────┬────┐
    │    │    │    │
   GPU1 GPU2 GPU3 GPU4
    │    │    │    │
    └────┼────┴────┘
         │
    Sincronizar gradientes
```

| Estrategia | Cómo Funciona |
|---|---|
| **DDP** | Cada GPU tiene copia completa del modelo, diferentes lotes de datos, sincroniza gradientes |
| **FSDP** | Fragmentar parámetros del modelo en GPUs, reconstruir durante adelante/atrás |
| **Paralelismo de Tensor** | Dividir el cómputo de una capa en múltiples GPUs |
| **Paralelismo de Pipeline** | Poner diferentes capas en diferentes GPUs |

FSDP es el más común para entrenar LLMs (divide parámetros, gradientes y estados del optimizador).

---

## 8. Errores Comunes

1. **Tasa de aprendizaje demasiado alta**: la falla de entrenamiento más común. Si la pérdida oscila o va a NaN, reducir la LR.

2. **No usar planificación de tasa de aprendizaje**: una LR constante rara vez es óptima. El decaimiento coseno o la reducción en meseta casi siempre son mejores.

3. **Tamaño de lote demasiado grande sin ajustar LR**: si duplicas el tamaño de lote, duplica la LR (regla de escalado lineal). De lo contrario, los gradientes son menos estocásticos y las actualizaciones son demasiado pequeñas.

4. **Olvidar el recorte de gradiente para RNNs/transformers**: estas arquitecturas son propensas a la explosión de gradientes. Recortar a norma 1.0 por defecto.

5. **No monitorear normas de gradiente**: observar solo la pérdida puede pasar por alto la inestabilidad. Registrar normas de gradiente para detectar problemas temprano.

---

## 9. Verifica tu Comprensión

1. Adam tiene dos parámetros de momentum ($\beta_1, \beta_2$). ¿Qué controla cada uno? ($\beta_1$: momentum de dirección del gradiente, $\beta_2$: momentum de magnitud del gradiente)

2. ¿Por qué AdamW separa el decaimiento de peso de la tasa de aprendizaje adaptativa? (Adam estándar con regularización L2 escala la regularización por la LR adaptativa, lo cual es incorrecto.)

3. Tienes un tamaño de lote de 32 pero solo caben 8 en tu GPU. ¿Qué haces? (Usar acumulación de gradiente — sumar gradientes sobre 4 micro-lotes.)

4. LayerNorm normaliza a través de la dimensión de características. BatchNorm normaliza a través de la dimensión de lote. ¿Cuándo se preferiría cada uno?

5. Un modelo de 1B de parámetros no cabe en una GPU. ¿Qué estrategia de entrenamiento distribuido usas?

---

## 10. Resumen

Entrenar una red neuronal efectivamente requiere elegir el optimizador correcto, la planificación de tasa de aprendizaje, la normalización y la regularización. Adam es el optimizador predeterminado. El decaimiento coseno con calentamiento lineal es la planificación predeterminada. LayerNorm/RMSNorm son estándar para transformers. El recorte de gradiente previene la explosión. El aprendizaje por transferencia ahorra datos y cómputo. Elige la combinación correcta y monitorea tanto la pérdida como las normas de gradiente durante todo el entrenamiento.

---

## 11. Dónde Ir Ahora

- [[Redes Neuronales]] — Qué entrenan estas técnicas
- [[Fine-tuning]] — Aplicar estas técnicas a modelos pre-entrenados
- [[Regularización]] — Prevenir sobreajuste durante el entrenamiento
