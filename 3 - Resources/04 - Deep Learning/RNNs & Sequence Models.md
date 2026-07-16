---
tags: [deep-learning, rnn, sequences, advanced]
status: growing
created: 2026-06-27
---

# RNNs y Modelos Secuenciales

## 1. Escenario de aprendizaje

Estás construyendo un modelo de traducción automática: convertir oraciones en inglés a español. El texto es secuencial — el orden de las palabras importa, y las oraciones pueden tener longitudes variables. Las RNNs fueron la primera arquitectura neuronal diseñada para manejar secuencias manteniendo un **estado oculto** que actúa como memoria, procesando cada palabra y actualizando el estado interno. Sin embargo, las RNNs tienen limitaciones graves con secuencias largas, lo que llevó a innovaciones como LSTMs, GRUs y finalmente Transformers.

---

## 2. La Red Neuronal Recurrente

### 2.1 La Idea Central

En lugar de procesar cada entrada independientemente, una RNN procesa tokens uno a la vez y mantiene un **estado oculto** $h_t$ que se pasa de un paso al siguiente:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

En cada paso, el estado oculto $h_t$ es una función de:
- La entrada actual $x_t$
- El estado oculto anterior $h_{t-1}$ (que codifica todo lo visto hasta ahora)

### 2.2 Desenrollamiento en el Tiempo

```
     y₁       y₂       y₃
     ↑        ↑        ↑
     h₁ →→→→ h₂ →→→→ h₃
     ↑        ↑        ↑
     x₁       x₂       x₃
```

Los mismos pesos ($W_{hh}, W_{xh}, b$) se usan en cada paso de tiempo. Esto es **compartición de pesos** — la misma transformación se aplica en cada posición, haciendo que las RNNs sean eficientes y capaces de manejar secuencias de longitud variable.

### 2.3 El Problema del Gradiente que Desaparece en RNNs

La propagación hacia atrás en el tiempo (BPTT) desenrolla la RNN y aplica propagación hacia atrás a través de todos los pasos de tiempo (ver [[Gradient-Based Optimization]]). El gradiente involucra un producto de muchas matrices Jacobianas:

$$\frac{\partial L}{\partial W} \propto \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}}$$

Para secuencias largas ($T$ grande), este producto tiende a:
- **Desaparecer** si los eigenvalores de $\frac{\partial h_t}{\partial h_{t-1}} < 1$ — no puede aprender dependencias de largo alcance
- **Explotar** si los eigenvalores $> 1$ — inestabilidad en el entrenamiento

Esta es la limitación fundamental de las RNNs vanilla: no pueden capturar dependencias más allá de ~10-20 pasos.

---

## 3. LSTM — Memoria a Largo y Corto Plazo

### 3.1 La Solución

Las LSTMs (1997) introdujeron **compuertas** para controlar el flujo de información. La innovación clave es el **estado de celda** $C_t$ — una autopista para la información que atraviesa la red directamente, con modificaciones mínimas.

### 3.2 Las Compuertas

**Compuerta de olvido**: qué descartar del pasado
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Compuerta de entrada**: qué información nueva almacenar
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidata**: información nueva a potencialmente añadir
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Actualización del estado de celda**: combinar olvido y adición
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Compuerta de salida**: qué revelar de la celda
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### 3.3 Intuición

Piensa en $C_t$ como una cinta transportadora de información. La compuerta de olvido decide qué tirar, la compuerta de entrada decide qué añadir, y la compuerta de salida decide qué revelar. La cinta transportadora ($C_t$) solo tiene operaciones lineales (sumar, multiplicar por compuertas), así que los gradientes fluyen a través de ella sin desaparecer.

Las LSTMs pueden recordar patrones cientos de pasos atrás — suficiente para la mayoría de tareas secuenciales prácticas.

---

## 4. GRU — Unidad Recurrente con Compuertas

Una LSTM simplificada con dos compuertas en lugar de tres:

- **Compuerta de reinicio**: cuánto del pasado olvidar
- **Compuerta de actualización**: cuánta información nueva usar

Menos parámetros que LSTM, rendimiento similar en la mayoría de tareas. A menudo preferida cuando los datos de entrenamiento son limitados.

---

## 5. RNNs Bidireccionales

Las RNNs estándar solo miran el contexto pasado. Las RNNs bidireccionales procesan la secuencia tanto hacia adelante como hacia atrás, luego concatenan los estados ocultos:

```python
h_forward = [h₁, h₂, h₃, ...]
h_backward = [... , h₃, h₂, h₁]  # procesado en reversa
h = concatenate(h_forward, h_backward)
```

Esto da a cada posición acceso a contexto tanto pasado como futuro. Esencial para tareas de clasificación y etiquetado de secuencias (análisis de sentimiento, NER).

---

## 6. El Mecanismo de Atención

### 6.1 El Problema con las RNNs

Una RNN debe comprimir toda la secuencia de entrada en un solo vector $h_T$ antes de generar la salida. La información de tokens tempranos se diluye.

### 6.2 Atención

La atención permite al decodificador **mirar hacia atrás** a todos los estados ocultos del codificador (enraizado en [[Probability]] — la distribución softmax sobre relevancias):

$$\text{Atención}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

En cada paso de decodificación, el modelo calcula una suma ponderada de todos los estados del codificador, donde los pesos reflejan la relevancia para la posición de decodificación actual.

La atención resolvió el problema de cuello de botella de las RNNs y abrió el camino para la arquitectura Transformer.

---

## 7. Alternativas Modernas

| Arquitectura | Fortalezas | Debilidades |
|---|---|---|
| **Transformers** | Paralelizable, dependencias de largo alcance, estado del arte | Memoria O(n²), costoso para secuencias largas |
| **CNNs para secuencias** (WaveNet, ConvS2S) | Paralelizable, estable | Receptive field limitado sin dilatación |
| **Mamba / Modelos de Espacio de Estado** | Memoria O(n), largo alcance, paralelizable | Nuevo, menos probado |

Para la mayoría de tareas secuenciales hoy, los Transformers son la opción por defecto. Las RNNs siguen usándose cuando la latencia es crítica (reconocimiento de voz en tiempo real) o los datos son extremadamente largos (genómica).

---

## 8. Errores Comunes

1. **Usar RNNs vanilla para secuencias largas**: no pueden aprender dependencias de largo alcance. Usar LSTM, GRU, o Transformer.

2. **No enmascarar padding**: si las secuencias tienen diferentes longitudes, las posiciones con padding deben ser enmascaradas en la función de pérdida. De lo contrario, el modelo aprende a predecir el token de padding.

3. **RNNs bidireccionales para generación**: no puedes usar información futura cuando generas de izquierda a derecha. Usar unidireccional para tareas autoregresivas.

4. **Ignorar el orden de secuencia en datos**: mezclar datos de series temporales destruye la estructura temporal. Siempre dividir series temporales cronológicamente.

5. **Entrenar RNNs con tasas de aprendizaje demasiado altas**: las RNNs son sensibles a gradientes inestables. Usar recorte de gradiente y tasas de aprendizaje más bajas.

---

## 9. Comprueba tu Conocimiento

1. Una RNN procesa la oración "I am learning deep learning". ¿Cuántas veces se usan los pesos de la RNN? (Una por palabra, mismos pesos cada vez — 5 veces.)

2. ¿Por qué las LSTMs resuelven el problema del gradiente que desaparece mientras que las RNNs vanilla no? (El estado de celda tiene flujo de gradiente lineal a través de la compuerta de olvido.)

3. Tienes una tarea de clasificación de sentimiento. ¿Usarías una RNN unidireccional o bidireccional? ¿Por qué?

4. La atención calcula una suma ponderada de estados del codificador. ¿Qué determina los pesos?

5. Un Transformer puede procesar todos los tokens en paralelo. ¿Por qué una RNN no puede hacer esto? (Dependencia secuencial del estado oculto anterior.)

---

## 10. Resumen

Las RNNs procesan secuencias manteniendo un estado oculto que actúa como memoria. Las LSTMs y GRUs resuelven el problema del gradiente que desaparece con mecanismos de compuertas, permitiendo dependencias de largo alcance. Las RNNs bidireccionales incorporan contexto futuro. La atención permite a los modelos mirar hacia atrás a toda la entrada, resolviendo el cuello de botella de información. Aunque los Transformers han reemplazado en gran medida a las RNNs para NLP, las RNNs siguen siendo relevantes para streaming, tiempo real y aplicaciones de secuencias extremadamente largas.

---

## 11. ¿Dónde ir Siguente?

- [[Transformers]] — La arquitectura que reemplazó a las RNNs en la mayoría de aplicaciones
- [[Neural Networks]] — Conceptos fundacionales
- [[Training Techniques]] — Recorte de gradiente y optimización para RNNs
