---
tags: [llm, transformers, advanced]
status: growing
created: 2026-06-27
---

# Arquitectura Transformer (LLMs Modernos)

## 1. Escenario de aprendizaje

Imagina que trabajas en una startup y necesitas elegir entre GPT-4, Llama 3 y Mistral para construir un asistente de código. Cada modelo tiene fortalezas distintas, pero todas comparten una arquitectura común. Para tomar una decisión informada — y no solo seguir tendencias — necesitas entender cómo funcionan por dentro. Este conocimiento te permite optimizar inferencia, elegir el modelo correcto para tu caso de uso y diagnosticar problemas cuando algo sale mal.

El artículo original de Transformer (2017) introdujo una arquitectura general para procesamiento de secuencias. Los LLMs modernos — GPT-4, Llama 3, Claude, Gemini — son descendientes directos, pero con mejoras arquitectónicas significativas basadas en investigación de [[Neural Networks]]. Entender estos detalles es esencial para cualquiera que trabaje con o construya LLMs.

Esta nota cubre la arquitectura **solo decoder** que impulsa los LLMs actuales, incluyendo los componentes específicos que difieren del Transformer original.

---

## 2. Del Transformer Original al LLM Moderno

### 2.1 El Cambio

| Componente | Transformer Original (2017) | LLM Moderno (2024) |
|---|---|---|
| Arquitectura | Encoder-Decoder | Solo decoder |
| Normalización | Post-LayerNorm | Pre-RMSNorm |
| Activación | ReLU | SwiGLU / GELU |
| Codificación posicional | Sinusoidal | RoPE |
| Atención | Multi-head completa | Grouped Query Attention (GQA) |
| FFN | MLP de 2 capas | FFN con compuerta (SwiGLU) |
| Tamaño de vocabulario | ~37K | 32K-128K |

### 2.2 ¿Por qué solo Decoder?

Encoder-decoder fue diseñado para traducción (una secuencia → otra). Para modelado de lenguaje y generación, una arquitectura solo decoder:
- Es más simple (una pila en lugar de dos)
- Escala mejor (todos los parámetros hacen generación)
- Funciona para aprendizaje en contexto (prompts + completaciones en el mismo formato)

---

## 3. El Bloque del LLM Moderno

Un bloque típico de Llama 3 / Mistral:

```
Input → RMSNorm → Grouped Query Attention → Add → RMSNorm → SwiGLU FFN → Add → Output
```

### 3.1 RMSNorm

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

Más simple que LayerNorm (sin resta de media). Más rápido y funciona igual de bien.

**Pre-normalización**: la normalización se aplica **antes** de cada subcapa (atención, FFN), no después. Esto hace que el entrenamiento sea más estable, especialmente en la inicialización.

### 3.2 Grouped Query Attention (GQA)

La atención multi-head estándar tiene $h$ cabezas de consulta, $h$ cabezas de clave, $h$ cabezas de valor. GQA usa menos cabezas de clave/valor que cabezas de consulta:

```
# Ejemplo: Llama 3 70B
Cabezas de consulta: 64
Cabezas de clave:    8
Cabezas de valor:    8
```

**Por qué**: la caché KV almacena una clave y un valor para cada cabeza por token. Con GQA, la caché es 8× más pequeña para el mismo número de cabezas de consulta. Esto es crítico para inferencia de contexto largo.

### 3.3 Activación SwiGLU

$$\text{SwiGLU}(x) = (x \cdot W_1) \odot \sigma(x \cdot W_3) \cdot W_2$$

Una variante con compuerta de Swish. La "compuerta" ($\sigma(xW_3)$) controla el flujo de información. Más expresiva que ReLU o GELU, con una pequeña sobrecarga computacional.

Para mantener el número de parámetros, la dimensión oculta del FFN se reduce (ej., 8/3 × d_model en lugar de 4 × d_model).

### 3.4 Rotary Position Embedding (RoPE)

En lugar de añadir codificación posicional a la entrada, RoPE **rota** los vectores de consulta y clave según su posición:

$$\text{RoPE}(x_m, m) = R(m) \cdot x_m$$

- $m$ es la posición
- $R(m)$ es una matriz de rotación
- El puntaje de atención $(Q_m)(K_n)^T$ codifica naturalmente la **posición relativa** $m-n$

**Ventajas**:
- Información de posición relativa (no absoluta)
- Decae con la distancia (los tokens cercanos tienen atención más fuerte)
- Puede extrapolar a secuencias más largas que las vistas durante el entrenamiento

---

## 4. Tokenización

### 4.1 Por qué Importa la Tokenización

El modelo no ve caracteres — ve tokens (subpalabras). El vocabulario y el algoritmo de tokenización determinan:
- Cuántos tokens se necesitan para codificar texto (afecta la longitud de secuencia)
- Cómo maneja el modelo palabras raras, números, puntuación
- La unidad "nativa" de procesamiento del modelo

### 4.2 Tokenizadores Comunes

| Tokenizador | Vocabulario | Usado Por |
|---|---|---|
| **BPE** (Byte-Pair Encoding) | 50K | GPT-2, GPT-4 |
| **SentencePiece** | 32K-128K | Llama, Mistral, T5 |
| **Tiktoken** | ~100K | OpenAI (GPT-4, o1) |
| **Unigram** | Variable | ALBERT, XLNet |

**BPE**: comienza con caracteres individuales, fusiona iterativamente el par más frecuente.

Ejemplo: "baj" + "o" → "bajo" (si "o" sigue a "baj" con suficiente frecuencia).

### 4.3 Desafíos de Tokenización en LLMs

- **Números**: "123" podría ser 1 token o 3. La aritmética es difícil porque el modelo ve "12" + "3" de manera diferente a "1" + "23".
- **Multilingüe**: algunos idiomas se tokenizan de manera mucho menos eficiente. El inglés obtiene ~1 token/palabra, el vietnamita podría obtener 3-4.
- **Tokens especiales**: `<|begin_of_text|>`, `<|end_of_text|>`, tokens de llamada a función, etc.

---

## 5. Leyes de Escalado y Entrenamiento

### 5.1 Leyes de Escalado Empíricas

$$L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

- La pérdida disminuye como una ley de potencia (un concepto clave en [[Statistics]]) con más parámetros ($N$) o más datos ($D$)
- $\alpha \approx 0.076, \beta \approx 0.103$ (del artículo Chinchilla)

### 5.2 El Hallazgo de Chinchilla

Para un presupuesto de cómputo dado, la relación óptima es ~20 tokens de datos de entrenamiento por parámetro:

```
Modelo 7B → 140B tokens
Modelo 70B → 1.4T tokens
```

Antes de Chinchilla (2022), los modelos estaban subentrenados (demasiados parámetros, muy pocos datos). Los LLMs modernos siguen (o superan) las relaciones óptimas de Chinchilla usando [[Training Techniques]] avanzadas.

### 5.3 Entrenamiento Óptimo en Cómputo

| Modelo | Parámetros | Tokens de Entrenamiento | Relación |
|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | 1.7:1 (subentrenado) |
| Llama 1 (2023) | 65B | 1.4T | 22:1 (Chinchilla) |
| Llama 3 (2024) | 70B | 15T | 214:1 (¡más datos!) |
| Chinchilla (2022) | 70B | 1.4T | 20:1 (óptimo) |

Llama 3 muestra que más datos continúa mejorando el rendimiento incluso más allá del óptimo de Chinchilla — pero las ganancias son decrecientes.

---

## 6. Inferencia

### 6.1 El Bucle de Generación

```
Input: "La capital de Francia es"
Paso 1: tokens = [La, capital, de, Francia, es]
Paso 2: calcular logits para el siguiente token
Paso 3: muestrear "París" (probabilidad 0.7) o greedy (siempre "París")
Paso 4: tokens = [..., es, París]
Paso 5: repetir hasta <EOS> o longitud máxima
```

### 6.2 Temperatura y Muestreo

- **Temperatura** $\tau$: escala los logits antes de softmax (una idea central en [[Probability]]). $\tau < 1$: más definido (más determinista). $\tau > 1$: más plano (más diverso).
- **Top-k**: muestrear solo de los $k$ tokens con mayor probabilidad.
- **Top-p (núcleo)**: muestrear del conjunto más pequeño de tokens cuya probabilidad acumulada > $p$.
- **Muestreo típico**: muestrear tokens cercanos a la probabilidad esperada (basado en entropía).

### 6.3 Caché KV — El Cuello de Botella de Memoria

Durante la generación, cada nuevo token recalcula la atención contra todos los tokens anteriores. La caché KV almacena los tensores K y V para todas las capas y tokens anteriores para evitar recálculo.

**Costo de memoria**: $2 \times \text{capas} \times \text{dim\_oculta} \times \text{long\_secuencia} \times \text{precisión} \times \text{num\_kv\_heads}$

Para Llama 3 70B con contexto de 8K: ~16GB para la caché KV por secuencia.

---

## 7. Common Mistakes

1. **Subestimar la memoria de inferencia**: la caché KV crece linealmente con la longitud de secuencia y el tamaño de lote. Un modelo que cabe en memoria durante el paso directo puede quedarse sin memoria durante la generación.

2. **Tokenizador incorrecto**: usar un tokenizador que no coincide con el tokenizador de entrenamiento del modelo produce salidas incoherentes. Usa siempre el tokenizador original del modelo.

3. **Lado de padding incorrecto para modelos solo decoder**: los modelos solo decoder deben tener padding a la IZQUIERDA (no a la derecha) para generación por lotes, así el modelo ve el contexto completo del prompt.

4. **Ignorar las leyes de escalado para tu propio entrenamiento**: entrenar un modelo pequeño con un conjunto de datos pequeño puede ser ineficiente en cómputo. Verifica la relación óptima.

---

## 8. Check Your Understanding

1. ¿Por qué GQA reduce el tamaño de la caché KV en comparación con la atención multi-head estándar? (Menos cabezas de clave/valor que cabezas de consulta.)

2. RoPE codifica posición RELATIVA, no absoluta. ¿Por qué es útil esto para contexto largo? (Las posiciones relativas están acotadas incluso cuando la longitud de secuencia crece.)

3. Un modelo de 7B entrenado con 140B tokens sigue el escalado de Chinchilla. ¿Qué sucede si lo entrenas con 1.4T tokens? (Mejor rendimiento, pero rendimientos decrecientes — 10× datos ≠ 10× mejora.)

4. ¿Por qué Pre-RMSNorm hace que el entrenamiento sea más estable que Post-LayerNorm? (Los gradientes fluyen más directamente a través de las ramas residuales.)

5. Generas texto con temperatura=0.1 vs temperatura=1.0. ¿Cuál es la diferencia? (0.1: casi determinista, 1.0: más diverso)

---

## 9. Resumen

Modern LLMs are decoder-only Transformers with key optimizations: RoPE for position, GQA for efficient inference, SwiGLU for expressiveness, and RMSNorm for stability. Tokenization converts text to tokens (subwords). Scaling laws guide how much data to train on. Inference uses autoregressive generation with a KV cache for efficiency. Understanding these components is essential for working with, fine-tuning, or deploying LLMs.

---

## 10. Where to Go Next

- [[Transformers]] — La arquitectura original
- [[Prompt Engineering]] — Usando LLMs efectivamente
- [[Fine-tuning]] — Adaptando LLMs a tareas personalizadas
