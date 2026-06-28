---
tags: [deep-learning, transformers, advanced]
status: growing
created: 2026-06-27
---

# Transformers

## 1. Why This Matters

The Transformer architecture (Vaswani et al., 2017) is the single most important architectural innovation in deep learning since backpropagation. It replaced RNNs for NLP, then extended to vision (ViT), audio (Whisper), speech (AudioLM), and multi-modal (CLIP, GPT-4V).

The core insight: **self-attention** lets every token directly attend to every other token, eliminating the sequential bottleneck of RNNs and enabling massive parallelization.

---

## 2. The Core Innovation: Self-Attention

### 2.1 The Problem with RNNs

In RNNs, information flows sequentially: token 1 → token 2 → token 3 → ... Token 100 must be processed through 99 steps before it can influence the output. This is slow (not parallelizable) and loses information over long distances.

### 2.2 Self-Attention

Self-attention computes **direct relationships between every pair of positions** in a single operation (see [[Linear Algebra]] for the underlying matrix operations):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step by step**:
1. Each token is projected into three vectors: **Query** (Q), **Key** (K), **Value** (V)
2. $QK^T$: compute dot products between every query and every key → "relevance scores"
3. Scale by $\sqrt{d_k}$: prevent softmax saturation for large dimensions
4. Softmax: convert scores into a [[Probability]] distribution (weights sum to 1 per token)
5. Multiply by $V$: take weighted sum of values

**Analogy**: think of a library. Q is your search query. K is the book's title/subject. V is the book's content. The dot product finds relevant books. Softmax decides how much to read each one. The weighted sum is the knowledge you take away.

### 2.3 Why Self-Attention Works

- **All-pair interactions**: each token can directly "see" every other token in one step
- **Parallelizable**: all dot products can be computed simultaneously (matrix multiplication)
- **No sequential bottleneck**: the path length between any two tokens is always 1

### 2.4 Multi-Head Attention

Instead of one attention function, use $h$ parallel heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$$

Each head learns a different type of relationship. In a language model, different heads might attend to syntax, semantics, coreference, position, etc.

---

## 3. The Transformer Block

Every Transformer block (layer) has the same structure:

```
Input → LayerNorm → Multi-Head Attention → Add (residual) → 
LayerNorm → FFN → Add (residual) → Output
```

**Feed-Forward Network (FFN)**:
$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

A two-layer MLP with expansion factor 4 (inner dimension = 4 × outer in most models). The FFN processes each token independently after the attention layer mixes information across tokens.

**Residual connections**: $x + F(x)$. Gradients flow directly through the skip connection, preventing vanishing gradients in deep stacks.

**Layer Normalization**: normalizes across features for each token independently. Stabilizes training.

---

## 4. Positional Encoding

Self-attention is **permutation invariant** — it treats the input as a set, not a sequence. "I ate the pizza" and "the pizza ate I" produce identical attention patterns without positional information.

**Solution**: add positional information to the input.

### 4.1 Sinusoidal (Original Transformer)

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

Each position gets a unique pattern of sine/cosine waves at different frequencies. The model can learn to use these to attend to specific positions.

### 4.2 Learned (BERT, GPT-2)

Let the model learn position embeddings during training. Simple and effective.

### 4.3 RoPE (Rotary Position Embedding)

Applied to queries and keys in attention, not added to input. Encodes relative position by rotating the Q/K vectors. Used in Llama, Mistral, and most modern LLMs.

---

## 5. Encoder vs Decoder

| Architecture | Usage | Examples |
|---|---|---|
| **Encoder-only** | Understanding (classification, NER, embeddings) | BERT, RoBERTa |
| **Decoder-only** | Generation (language modeling, text completion) | GPT, Llama, Claude |
| **Encoder-Decoder** | Seq2Seq (translation, summarization) | T5, BART |

**Decoder-only** (modern LLMs): each token can only attend to previous tokens (causal/masked attention). This enables autoregressive generation.

**Encoder**: bidirectional attention (each token sees all tokens). Better for understanding tasks.

---

## 6. Efficiency

### 6.1 The Quadratic Problem

Self-attention is O(n²) in both compute and memory because each of the $n$ tokens attends to all $n$ tokens. For long sequences, this becomes prohibitive.

### 6.2 FlashAttention

Reorganizes attention computation to avoid materializing the full $n \times n$ attention matrix in GPU memory. Uses tiling and kernel fusion. 2-4× faster, less memory, exact attention (not approximate).

### 6.3 KV Cache

During autoregressive generation, the model computes Q, K, V for each new token. K and V from previous tokens are **cached** and reused — only the new token's Q, K, V are computed. Without caching, generation would be O(n²) per token.

### 6.4 Grouped Query Attention (GQA)

Multiple query heads share fewer key/value heads. Used in Llama 2/3. Reduces KV cache size by 4-8× with minimal quality loss.

---

## 7. Common Mistakes

1. **Not scaling attention scores**: without the $\sqrt{d_k}$ scaling, softmax saturates for large dimensions, producing near-one-hot attention. All tokens attend to one token.

2. **Forgetting causal masking in decoder**: during training, the decoder must not see future tokens. Apply a causal mask (upper triangular of -inf).

3. **Not understanding the KV cache**: autoregressive generation with attention recalculates K and V for every previous token at each step — unless cached. Without caching, generation is $O(n^3)$.

4. **Positional encoding for non-sequential data**: if input order is not meaningful, you may not need positional encoding (e.g., set prediction).

---

## 8. Check Your Understanding

1. Self-attention is O(n²). If a sequence of length 100 takes 1ms, approximately how long will a sequence of length 500 take? (25ms — 5² × 1ms)

2. Why must decoder-only models use causal (masked) attention? (Each token should only see previous tokens for autoregressive generation.)

3. What problem does the KV cache solve, and why does it not apply during training? (Generation repeats previous computations; training sees all tokens at once.)

4. BERT uses bidirectional attention. GPT uses causal attention. When would you use each?

5. A transformer with 12 heads and d_model=768 has each head working in what dimension? (768/12 = 64)

---

## 9. Summary

Transformers replaced RNNs by replacing sequential processing with parallel self-attention. Self-attention lets every token directly attend to every other token, solving the long-range dependency problem. Multiple heads learn different relationship types. Positional encoding adds order information. The decoder-only variant (causal attention) enables autoregressive generation and powers modern LLMs.

---

## 10. Where to Go Next

- [[Transformer Architecture]] — Modern LLM architecture details (RoPE, SwiGLU, GQA)
- [[Neural Networks]] — Foundational concepts
- [[RNNs & Sequence Models]] — The architecture Transformers replaced
- [[Training Techniques]] — Optimizing transformer training
