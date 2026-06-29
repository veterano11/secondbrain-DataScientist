---
tags: [deep-learning, rnn, sequences, advanced]
status: growing
created: 2026-06-27
---

# RNNs & Sequence Models

## 1. Escenario de aprendizaje

Estás construyendo un modelo de traducción automática: convertir oraciones en inglés a español. El texto es secuencial — el orden de las palabras importa, y las oraciones pueden tener longitudes variables. Las RNNs fueron la primera arquitectura neuronal diseñada para manejar secuencias manteniendo un **estado oculto** que actúa como memoria, procesando cada palabra y actualizando el estado interno. Sin embargo, las RNNs tienen limitaciones graves con secuencias largas, lo que llevó a innovaciones como LSTMs, GRUs y finalmente Transformers.

---

## 2. The Recurrent Neural Network

### 2.1 The Core Idea

Instead of processing each input independently, an RNN processes tokens one at a time and maintains a **hidden state** $h_t$ that is passed from one step to the next:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

At each step, the hidden state $h_t$ is a function of:
- The current input $x_t$
- The previous hidden state $h_{t-1}$ (which encodes everything seen so far)

### 2.2 Unrolling in Time

```
     y₁       y₂       y₃
     ↑        ↑        ↑
     h₁ →→→→ h₂ →→→→ h₃
     ↑        ↑        ↑
     x₁       x₂       x₃
```

The same weights ($W_{hh}, W_{xh}, b$) are used at every time step. This is **weight sharing** — the same transformation is applied at each position, making RNNs efficient and able to handle variable-length sequences.

### 2.3 The Vanishing Gradient Problem in RNNs

Backpropagation through time (BPTT) unrolls the RNN and applies backpropagation across all time steps (see [[Gradient-Based Optimization]]). The gradient involves a product of many Jacobian matrices:

$$\frac{\partial L}{\partial W} \propto \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}}$$

For long sequences ($T$ large), this product tends to:
- **Vanish** if eigenvalues of $\frac{\partial h_t}{\partial h_{t-1}} < 1$ — cannot learn long-range dependencies
- **Explode** if eigenvalues $> 1$ — training instability

This is the fundamental limitation of vanilla RNNs: they cannot capture dependencies beyond ~10-20 steps.

---

## 3. LSTM — Long Short-Term Memory

### 3.1 The Solution

LSTMs (1997) introduced **gates** to control the flow of information. The key innovation is the **cell state** $C_t$ — a highway for information that runs straight through the network, with minimal modification.

### 3.2 The Gates

**Forget gate**: what to discard from the past
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input gate**: what new information to store
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate**: new information to potentially add
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell state update**: combine forgetting and adding
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output gate**: what to output from the cell
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### 3.3 Intuition

Think of $C_t$ as a conveyor belt of information. The forget gate decides what to throw away, the input gate decides what to add, and the output gate decides what to reveal. The conveyor belt ($C_t$) only has linear operations (add, multiply by gates), so gradients flow through it without vanishing.

LSTMs can remember patterns hundreds of steps back — enough for most practical sequence tasks.

---

## 4. GRU — Gated Recurrent Unit

A simplified LSTM with two gates instead of three:

- **Reset gate**: how much of the past to forget
- **Update gate**: how much of the new information to use

Fewer parameters than LSTM, similar performance in most tasks. Often preferred when training data is limited.

---

## 5. Bidirectional RNNs

Standard RNNs only look at past context. Bidirectional RNNs process the sequence both forward and backward, then concatenate the hidden states:

```python
h_forward = [h₁, h₂, h₃, ...]
h_backward = [... , h₃, h₂, h₁]  # processed in reverse
h = concatenate(h_forward, h_backward)
```

This gives each position access to both past and future context. Essential for classification and sequence labeling tasks (sentiment analysis, NER).

---

## 6. The Attention Mechanism

### 6.1 The Problem with RNNs

An RNN must compress the entire input sequence into a single vector $h_T$ before generating the output. Information from early tokens gets diluted.

### 6.2 Attention

Attention allows the decoder to **look back** at all encoder hidden states (rooted in [[Probability]] — the softmax distribution over relevances):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

At each decoding step, the model computes a weighted sum of all encoder states, where the weights reflect relevance to the current decoding position.

Attention solved the bottleneck problem of RNNs and paved the way for the Transformer architecture.

---

## 7. Modern Alternatives

| Architecture | Strengths | Weaknesses |
|---|---|---|
| **Transformers** | Parallelizable, long-range dependencies, state-of-the-art | O(n²) memory, expensive for long sequences |
| **CNNs for sequences** (WaveNet, ConvS2S) | Parallelizable, stable | Limited receptive field without dilation |
| **Mamba / State Space Models** | O(n) memory, long-range, parallelizable | New, less proven |

For most sequence tasks today, Transformers are the default. RNNs are still used when latency is critical (real-time speech recognition) or data is extremely long (genomics).

---

## 8. Common Mistakes

1. **Using vanilla RNNs for long sequences**: they cannot learn long-range dependencies. Use LSTM, GRU, or Transformer.

2. **Not masking padding**: if sequences have different lengths, padded positions must be masked in the loss function. Otherwise the model learns to predict the padding token.

3. **Bidirectional RNNs for generation**: you cannot use future information when generating left-to-right. Use unidirectional for autoregressive tasks.

4. **Ignoring sequence order in data**: shuffling time series data destroys the temporal structure. Always split time series chronologically.

5. **Training RNNs with too large learning rates**: RNNs are sensitive to unstable gradients. Use gradient clipping and lower learning rates.

---

## 9. Check Your Understanding

1. An RNN processes the sentence "I am learning deep learning". How many times are the RNN weights used? (One per word, same weights each time — 5 times.)

2. Why do LSTMs solve the vanishing gradient problem while vanilla RNNs do not? (The cell state has linear gradient flow through the forget gate.)

3. You have a sentiment classification task. Would you use a unidirectional or bidirectional RNN? Why?

4. Attention computes a weighted sum of encoder states. What determines the weights?

5. A Transformer can process all tokens in parallel. Why can an RNN not do this? (Sequential dependence on previous hidden state.)

---

## 10. Resumen

RNNs process sequences by maintaining a hidden state that acts as memory. LSTMs and GRUs solve the vanishing gradient problem with gating mechanisms, enabling long-range dependencies. Bidirectional RNNs incorporate future context. Attention allows models to look back at the entire input, solving the information bottleneck. While Transformers have largely replaced RNNs for NLP, RNNs remain relevant for streaming, real-time, and extremely long sequence applications.

---

## 11. Where to Go Next

- [[Transformers]] — The architecture that replaced RNNs in most applications
- [[Neural Networks]] — Foundational concepts
- [[Training Techniques]] — Gradient clipping and optimization for RNNs
