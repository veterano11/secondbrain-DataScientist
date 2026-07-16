---
tags: [deep-learning, transformers, advanced]
status: growing
created: 2026-06-27
---

# Transformers

## 1. Escenario de aprendizaje

Quieres construir un asistente de chat que entienda y genere texto como un humano. Antes de Transformers, los modelos de lenguaje procesaban palabras una por una (como las RNNs), lo que era lento y perdía contexto en oraciones largas. La arquitectura Transformer (Vaswani et al., 2017) es la innovación arquitectónica más importante en deep learning desde backpropagation. Reemplazó las RNNs para NLP, luego se extendió a visión (ViT), audio (Whisper), habla (AudioLM) y multimodal (CLIP, GPT-4V).

La idea central: **self-attention** permite que cada token atienda directamente a todos los demás tokens, eliminando el cuello de botella secuencial de las RNNs y habilitando la paralelización masiva.

---

## 2. La Innovación Central: Self-Attention

### 2.1 El Problema con las RNNs

En las RNNs, la información fluye secuencialmente: token 1 → token 2 → token 3 → ... El token 100 debe ser procesado a través de 99 pasos antes de poder influir en la salida. Esto es lento (no paralelizable) y pierde información a largas distancias.

### 2.2 Self-Attention

El self-attention calcula **relaciones directas entre cada par de posiciones** en una sola operación (ver [[Linear Algebra]] para las operaciones matriciales subyacentes):

$$\text{Atención}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Paso a paso**:
1. Cada token se proyecta en tres vectores: **Query** (Q), **Key** (K), **Value** (V)
2. $QK^T$: calcular productos punto entre cada query y cada key → "puntuaciones de relevancia"
3. Escalar por $\sqrt{d_k}$: prevenir saturación del softmax para dimensiones grandes
4. Softmax: convertir puntuaciones en una distribución de [[Probability]] (los pesos suman 1 por token)
5. Multiplicar por $V$: tomar la suma ponderada de valores

**Analogía**: piensa en una biblioteca. Q es tu consulta de búsqueda. K es el título/tema del libro. V es el contenido del libro. El producto punto encuentra libros relevantes. Softmax decide cuánto leer de cada uno. La suma ponderada es el conocimiento que te llevas.

### 2.3 Por Qué Funciona el Self-Attention

- **Interacciones de todos los pares**: cada token puede "ver" directamente a cada otro token en un paso
- **Paralelizable**: todos los productos punto pueden calcularse simultáneamente (multiplicación de matrices)
- **Sin cuello de botella secuencial**: la longitud del camino entre cualquier par de tokens siempre es 1

### 2.4 Atención Multi-Cabezal

En lugar de una función de atención, usar $h$ cabezas paralelas:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$$

Cada cabeza aprende un tipo diferente de relación. En un modelo de lenguaje, diferentes cabezas pueden atender a sintaxis, semántica, coreferencia, posición, etc.

---

## 3. El Bloque Transformer

Cada bloque Transformer (capa) tiene la misma estructura:

```
Entrada → LayerNorm → Atención Multi-Cabezal → Add (residual) → 
LayerNorm → FFN → Add (residual) → Salida
```

**Red Feed-Forward (FFN)**:
$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

Un MLP de dos capas con factor de expansión 4 (dimensión interna = 4 × externa en la mayoría de modelos). La FFN procesa cada token independientemente después de que la capa de atención mezcla información entre tokens.

**Conexiones residuales**: $x + F(x)$. Los gradientes fluyen directamente a través de la conexión skip, previniendo gradientes que desaparecen en apilamientos profundos.

**Normalización por Capas**: normaliza a través de features para cada token independientemente. Estabiliza el entrenamiento.

---

## 4. Codificación Posicional

El self-attention es **invariante a permutaciones** — trata la entrada como un conjunto, no como una secuencia. "I ate the pizza" y "the pizza ate I" producen patrones de atención idénticos sin información posicional.

**Solución**: añadir información posicional a la entrada.

### 4.1 Sinusoidal (Transformer Original)

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

Cada posición obtiene un patrón único de ondas seno/coseno en diferentes frecuencias. El modelo puede aprender a usarlas para atender a posiciones específicas.

### 4.2 Aprendido (BERT, GPT-2)

Dejar que el modelo aprenda embeddings posicionales durante el entrenamiento. Simple y efectivo.

### 4.3 RoPE (Embedding Posicional Rotatorio)

Se aplica a queries y keys en la atención, no se añade a la entrada. Codifica la posición relativa rotando los vectores Q/K. Usado en Llama, Mistral y la mayoría de LLMs modernos.

---

## 5. Codificador vs Decodificador

| Arquitectura | Uso | Ejemplos |
|---|---|---|
| **Solo codificador** | Comprensión (clasificación, NER, embeddings) | BERT, RoBERTa |
| **Solo decodificador** | Generación (modelado de lenguaje, completado de texto) | GPT, Llama, Claude |
| **Codificador-Decodificador** | Seq2Seq (traducción, resumen) | T5, BART |

**Solo decodificador** (LLMs modernos): cada token solo puede atender a tokens anteriores (atención causal/enmascarada). Esto habilita la generación autoregresiva.

**Codificador**: atención bidireccional (cada token ve todos los tokens). Mejor para tareas de comprensión.

---

## 6. Eficiencia

### 6.1 El Problema Cuadrático

El self-attention es O(n²) tanto en cómputo como en memoria porque cada uno de los $n$ tokens atiende a todos los $n$ tokens. Para secuencias largas, esto se vuelve prohibitivo.

### 6.2 FlashAttention

Reorganiza el cómputo de atención para evitar materializar la matriz de atención completa $n \times n$ en la memoria de la GPU. Usa tiling y fusión de kernels. 2-4× más rápido, menos memoria, atención exacta (no aproximada).

### 6.3 KV Cache

Durante la generación autoregresiva, el modelo calcula Q, K, V para cada token nuevo. K y V de tokens anteriores se **almacenan en caché** y reutilizan — solo se calculan Q, K, V del token nuevo. Sin caché, la generación sería O(n²) por token.

### 6.4 Atención de Consultas Agrupadas (GQA)

Múltiples cabezas de consulta comparten menos cabezas de key/value. Usado en Llama 2/3. Reduce el tamaño del KV cache en 4-8× con pérdida mínima de calidad.

---

## 7. Errores Comunes

1. **No escalar puntuaciones de atención**: sin el escalado $\sqrt{d_k}$, el softmax satura para dimensiones grandes, produciendo atención casi one-hot. Todos los tokens atienden a un solo token.

2. **Olvidar el enmascaramiento causal en el decodificador**: durante el entrenamiento, el decodificador no debe ver tokens futuros. Aplicar una máscara causal (triangular superior de -inf).

3. **No entender el KV cache**: la generación autoregresiva con atención recalcula K y V para cada token anterior en cada paso — a menos que se almacenen en caché. Sin caché, la generación es $O(n^3)$.

4. **Codificación posicional para datos no secuenciales**: si el orden de entrada no es significativo, es posible que no necesites codificación posicional (ej: predicción de conjuntos).

---

## 8. Comprueba tu Conocimiento

1. El self-attention es O(n²). Si una secuencia de longitud 100 toma 1ms, aproximadamente cuánto tomará una secuencia de longitud 500? (25ms — 5² × 1ms)

2. ¿Por qué los modelos solo decodificador deben usar atención causal (enmascarada)? (Cada token solo debe ver tokens anteriores para generación autoregresiva.)

3. ¿Qué problema resuelve el KV cache y por qué no aplica durante el entrenamiento? (La generación repite cálculos anteriores; el entrenamiento ve todos los tokens a la vez.)

4. BERT usa atención bidireccional. GPT usa atención causal. ¿Cuándo usarías cada uno?

5. Un transformer con 12 cabezas y d_model=768 tiene cada cabeza trabajando en qué dimensión? (768/12 = 64)

---

## 9. Resumen

Los Transformers reemplazaron a las RNNs reemplazando el procesamiento secuencial con self-attention paralelo. El self-attention permite que cada token atienda directamente a cada otro token, resolviendo el problema de dependencias de largo alcance. Múltiples cabezas aprenden diferentes tipos de relaciones. La codificación posicional añade información de orden. La variante solo decodificador (atención causal) habilita la generación autoregresiva y potencia los LLMs modernos.

---

## 10. ¿Dónde ir Siguente?

- [[Transformer Architecture]] — Detalles de arquitectura de LLMs modernos (RoPE, SwiGLU, GQA)
- [[Neural Networks]] — Conceptos fundacionales
- [[RNNs & Sequence Models]] — La arquitectura que los Transformers reemplazaron
- [[Training Techniques]] — Optimización del entrenamiento de transformers
