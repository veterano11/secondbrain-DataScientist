---
tags: [llm, transformers, advanced]
status: growing
created: 2026-06-27
---

# Transformer Architecture (Modern LLMs)

## 1. Why This Matters

The original Transformer paper (2017) introduced a general architecture for sequence processing. Modern LLMs — GPT-4, Llama 3, Claude, Gemini — are direct descendants, but with significant architectural improvements rooted in [[Neural Networks]] research. Understanding these details is essential for anyone working with or building LLMs.

This note covers the **decoder-only** architecture that powers today's LLMs, including the specific components that differ from the original Transformer.

---

## 2. From Original Transformer to Modern LLM

### 2.1 The Shift

| Component | Original Transformer (2017) | Modern LLM (2024) |
|---|---|---|
| Architecture | Encoder-Decoder | Decoder-only |
| Normalization | Post-LayerNorm | Pre-RMSNorm |
| Activation | ReLU | SwiGLU / GELU |
| Position encoding | Sinusoidal | RoPE |
| Attention | Full multi-head | Grouped Query Attention (GQA) |
| FFN | 2-layer MLP | Gated FFN (SwiGLU) |
| Vocab size | ~37K | 32K-128K |

### 2.2 Why Decoder-Only?

Encoder-decoder was designed for translation (one sequence → another). For language modeling and generation, a decoder-only architecture:
- Is simpler (one stack instead of two)
- Scales better (all parameters do generation)
- Works for in-context learning (prompts + completions in the same format)

---

## 3. The Modern LLM Block

A typical Llama 3 / Mistral block:

```
Input → RMSNorm → Grouped Query Attention → Add → RMSNorm → SwiGLU FFN → Add → Output
```

### 3.1 RMSNorm

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

Simpler than LayerNorm (no mean subtraction). Faster and works just as well.

**Pre-normalization**: normalization is applied **before** each sublayer (attention, FFN), not after. This makes training more stable, especially at initialization.

### 3.2 Grouped Query Attention (GQA)

Standard multi-head attention has $h$ query heads, $h$ key heads, $h$ value heads. GQA uses fewer key/value heads than query heads:

```
# Example: Llama 3 70B
Query heads: 64
Key heads:    8
Value heads:  8
```

**Why**: the KV cache stores a key and value for each head per token. With GQA, the cache is 8× smaller for the same number of query heads. This is critical for long-context inference.

### 3.3 SwiGLU Activation

$$\text{SwiGLU}(x) = (x \cdot W_1) \odot \sigma(x \cdot W_3) \cdot W_2$$

A gated variant of Swish. The "gate" ($\sigma(xW_3)$) controls information flow. More expressive than ReLU or GELU, with a small computational overhead.

To maintain parameter count, the FFN hidden dimension is reduced (e.g., 8/3 × d_model instead of 4 × d_model).

### 3.4 Rotary Position Embedding (RoPE)

Instead of adding positional encoding to the input, RoPE **rotates** query and key vectors based on their position:

$$\text{RoPE}(x_m, m) = R(m) \cdot x_m$$

- $m$ is the position
- $R(m)$ is a rotation matrix
- Attention score $(Q_m)(K_n)^T$ naturally encodes **relative position** $m-n$

**Advantages**:
- Relative position information (not absolute)
- Decays with distance (nearby tokens have stronger attention)
- Can extrapolate to longer sequences than seen during training

---

## 4. Tokenization

### 4.1 Why Tokenization Matters

The model does not see characters — it sees tokens (subwords). The vocabulary and tokenization algorithm determine:
- How many tokens are needed to encode text (affects sequence length)
- How the model handles rare words, numbers, punctuation
- The model's "native" unit of processing

### 4.2 Common Tokenizers

| Tokenizer | Vocabulary | Used By |
|---|---|---|
| **BPE** (Byte-Pair Encoding) | 50K | GPT-2, GPT-4 |
| **SentencePiece** | 32K-128K | Llama, Mistral, T5 |
| **Tiktoken** | ~100K | OpenAI (GPT-4, o1) |
| **Unigram** | Variable | ALBERT, XLNet |

**BPE**: starts with individual characters, iteratively merges the most frequent pair.

Example: "low" + "er" → "lower" (if "er" follows "low" frequently enough).

### 4.3 LLM Tokenization Challenges

- **Numbers**: "123" might be 1 token or 3. Arithmetic is hard because the model sees "12" + "3" differently than "1" + "23".
- **Multilingual**: some languages are tokenized much less efficiently. English gets ~1 token/word, Vietnamese might get 3-4.
- **Special tokens**: `<|begin_of_text|>`, `<|end_of_text|>`, function call tokens, etc.

---

## 5. Scaling Laws and Training

### 5.1 Empirical Scaling Laws

$$L(N, D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

- Loss decreases as a power-law (a key concept in [[Statistics]]) with more parameters ($N$) or more data ($D$)
- $\alpha \approx 0.076, \beta \approx 0.103$ (from Chinchilla paper)

### 5.2 The Chinchilla Finding

For a given compute budget, the optimal ratio is ~20 tokens of training data per parameter:

```
7B model → 140B tokens
70B model → 1.4T tokens
```

Before Chinchilla (2022), models were undertrained (too many parameters, too little data). Modern LLMs follow (or exceed) Chinchilla-optimal ratios using advanced [[Training Techniques]].

### 5.3 Compute-Optimal Training

| Model | Parameters | Training Tokens | Ratio |
|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | 1.7:1 (undertrained) |
| Llama 1 (2023) | 65B | 1.4T | 22:1 (Chinchilla) |
| Llama 3 (2024) | 70B | 15T | 214:1 (more data!) |
| Chinchilla (2022) | 70B | 1.4T | 20:1 (optimal) |

Llama 3 shows that more data continues to improve performance even beyond Chinchilla-optimal — but the gains are diminishing.

---

## 6. Inference

### 6.1 The Generation Loop

```
Input: "The capital of France is"
Step 1: tokens = [The, capital, of, France, is]
Step 2: compute logits for next token
Step 3: sample "Paris" (probability 0.7) or greedy (always "Paris")
Step 4: tokens = [..., is, Paris]
Step 5: repeat until <EOS> or max length
```

### 6.2 Temperature and Sampling

- **Temperature** $\tau$: scale logits before softmax (a core idea in [[Probability]]). $\tau < 1$: sharper (more deterministic). $\tau > 1$: flatter (more diverse).
- **Top-k**: sample only from the $k$ highest probability tokens.
- **Top-p (nucleus)**: sample from the smallest set of tokens whose cumulative probability > $p$.
- **Typical sampling**: sample tokens near the expected probability (entropy-based).

### 6.3 KV Cache — The Memory Bottleneck

During generation, each new token recomputes attention against all previous tokens. The KV cache stores K and V tensors for all previous layers and tokens to avoid recomputation.

**Memory cost**: $2 \times \text{layers} \times \text{hidden\_dim} \times \text{sequence\_length} \times \text{precision} \times \text{num\_kv\_heads}$

For Llama 3 70B with 8K context: ~16GB for the KV cache per sequence.

---

## 7. Common Mistakes

1. **Underestimating inference memory**: the KV cache grows linearly with sequence length and batch size. A model that fits in memory during forward pass may OOM during generation.

2. **Tokenizer mismatch**: using a tokenizer that does not match the model's training tokenizer produces garbled outputs. Always use the model's original tokenizer.

3. **Wrong padding side for decoder-only models**: decoder-only models should be padded on the LEFT (not right) for batch generation, so the model sees the full prompt context.

4. **Ignoring scaling laws for your own training**: training a small model on a small dataset may be compute-inefficient. Check the optimal ratio.

---

## 8. Check Your Understanding

1. Why does GQA reduce KV cache size compared to standard multi-head attention? (Fewer key/value heads than query heads.)

2. RoPE encodes RELATIVE position, not absolute. Why is this useful for long context? (Relative positions are bounded even as sequence length grows.)

3. A 7B model trained on 140B tokens follows Chinchilla scaling. What happens if you train it on 1.4T tokens? (Better performance, but diminishing returns — 10× data ≠ 10× improvement.)

4. Why does pre-RMSNorm make training more stable than post-LayerNorm? (Gradients flow more directly through residual branches.)

5. You generate text with temperature=0.1 vs temperature=1.0. What is the difference? (0.1: nearly deterministic, 1.0: more diverse)

---

## 9. Summary

Modern LLMs are decoder-only Transformers with key optimizations: RoPE for position, GQA for efficient inference, SwiGLU for expressiveness, and RMSNorm for stability. Tokenization converts text to tokens (subwords). Scaling laws guide how much data to train on. Inference uses autoregressive generation with a KV cache for efficiency. Understanding these components is essential for working with, fine-tuning, or deploying LLMs.

---

## 10. Where to Go Next

- [[Transformers]] — The original architecture
- [[Prompt Engineering]] — Using LLMs effectively
- [[Fine-tuning]] — Adapting LLMs to custom tasks
