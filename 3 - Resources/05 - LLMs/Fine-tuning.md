---
tags: [llm, fine-tuning, advanced]
status: growing
created: 2026-06-27
---

# Fine-tuning

## 1. Why This Matters

Pre-trained LLMs are generalists — they know a bit about everything but excel at nothing specific. Fine-tuning adapts them to a particular domain, task, or behavior — the core idea behind [[Transfer Learning]]. It is how you turn a generic model into a specialized tool.

Fine-tuning is the difference between a model that "knows about" your codebase and one that "writes code in your style." Between a model that "understands" medical terminology and one that "diagnoses" from radiology reports correctly.

---

## 2. The Fine-Tuning Spectrum

### 2.1 Full Fine-Tuning

Update all parameters. Powerful but expensive.

**Memory**:
- Model weights: 70B × 2 bytes (BF16) = 140 GB
- Optimizer states (Adam): 70B × 4 × 4 bytes = 1.12 TB
- Gradients: 70B × 4 bytes = 280 GB
- **Total**: ~1.5 TB per GPU — impossible. Distributed across GPUs.

**When to use**: large target dataset, high-quality data, sufficient compute budget.

### 2.2 Parameter-Efficient Fine-Tuning (PEFT)

Update a tiny fraction of parameters. Almost as good as full fine-tuning for most tasks.

#### LoRA (Low-Rank Adaptation)

$$W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

**Step by step**:
1. Freeze the original weight matrix $W$
2. Add two small matrices $A$ and $B$ with rank $r$ (typically 8-64)
3. Only $A$ and $B$ are trained
4. After training, $BA$ can be merged into $W$ (zero inference overhead)

**Why it works**: the weight updates during fine-tuning have low "intrinsic rank" — the actual changes lie in a low-dimensional subspace.

**Memory**: a 7B model with LoRA trains on a single GPU with 24GB memory.

#### QLoRA

LoRA + 4-bit quantization of the base model. Enables fine-tuning a 70B model on a single 48GB GPU.

**Techniques**:
- NF4 quantization (normal float 4-bit)
- Double quantization (quantize the quantization constants)
- Paged optimizers (CPU offloading when GPU memory is exceeded)

#### Adapters

Insert small bottleneck layers between transformer blocks:

```
Input → Adapter(down) → ReLU → Adapter(up) → Output
       d → r              r → d
```

Only the adapter parameters are trained. Less efficient than LoRA (adds inference latency).

#### Prefix Tuning

Learn "virtual tokens" prepended to each layer's key/value. No new weights added, just learned embeddings.

---

## 3. Instruction Fine-Tuning

### 3.1 The Format

```
{
  "instruction": "Translate to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

The model learns to follow instructions in the format seen during training. This is how base models become chat/assistant models.

### 3.2 Chat Template

Models use specific chat templates that structure the conversation:

```python
# Llama 3 chat template
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is ML?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Machine learning is...
```

Always use the model's correct chat template. Using the wrong template degrades performance significantly.

---

## 4. Data Quality

### 4.1 Quantity vs Quality

100 high-quality examples > 10,000 noisy examples.

**Quality criteria**:
- **Correct**: the output is factually correct
- **Consistent**: follows the expected format and style
- **Diverse**: covers the range of inputs the model will see
- **Non-toxic**: no harmful content in outputs

### 4.2 Data Preparation

1. **Collect**: gather examples of the desired behavior
2. **Clean**: remove duplicates, fix formatting, check correctness
3. **Format**: apply the appropriate chat template
4. **Split**: train (90%), validation (10%)
5. **Deduplicate**: remove near-duplicates (LLM-based or embedding similarity)

---

## 5. RLHF — Reinforcement Learning from Human Feedback

### 5.1 The Problem

Fine-tuning on instructions teaches the model WHAT to do, but not HOW to behave. [[RLHF & Preference Optimization]] aligns the model with human preferences.

### 5.2 The Three Steps

**Step 1: Supervised Fine-Tuning (SFT)**
- Fine-tune on high-quality human demonstrations — a form of [[Supervised Learning]]
- Teaches the model the desired format and style

**Step 2: Reward Model Training**
- For each prompt, generate multiple outputs from the SFT model
- Humans rank the outputs (pairwise comparisons)
- Train a reward model to predict human preferences

**Step 3: PPO (Proximal Policy Optimization)**
- Use the reward model to score the LLM's outputs
- Optimize the LLM to maximize reward
- Add KL penalty to prevent the model from diverging too far from the SFT model

### 5.3 Alternatives to RLHF

| Method | Description | Pros | Cons |
|---|---|---|---|
| **DPO** | Direct Preference Optimization | Simpler, no reward model | May not scale as well |
| **ORPO** | Combined SFT + alignment | Single stage | New, less tested |
| **KTO** | Kahneman-Tversky Optimization | Only needs binary feedback | Less fine-grained |

DPO is the most popular alternative: it directly optimizes the policy on preference pairs without training a separate reward model.

---

## 6. Practical Checklist

```
☐ Data: min 100 high-quality examples
☐ Format: correct chat template
☐ Rank r: 8-64 (higher for more diverse tasks)
☐ Target modules: q_proj, v_proj (common), or all linear layers
☐ LR: 1e-4 to 5e-4 (LoRA), 1e-5 to 5e-5 (full)
☐ Batch size: gradient accumulation to 64-128 samples
☐ Epochs: 1-3 (more = overfitting risk)
☐ Evaluation: held-out validation set
☐ Monitor: loss curves, generation quality on validation — use [[Experiment Tracking]] to log these metrics
```

---

## 7. Common Mistakes

1. **Too many epochs**: LLMs overfit quickly. 1-3 epochs is usually enough. More epochs hurt generalization.

2. **Mismatched chat template**: using the wrong template causes garbled outputs. Verify that the format matches exactly.

3. **Forgetting to merge LoRA weights for deployment**: LoRA weights must be merged or loaded separately at inference time.

4. **Low-quality data**: garbage in, garbage out. Fine-tuning amplifies patterns in the training data — including errors.

5. **Not evaluating before/after**: if you cannot measure improvement, you do not know if fine-tuning helped. Always evaluate on a held-out test set.

---

## 8. Check Your Understanding

1. LoRA rank r=8 uses what fraction of a 4096×4096 weight matrix? (8×4096 + 4096×8 = 65K out of 4096² ≈ 16.8M → 0.4%)

2. Why does QLoRA enable fine-tuning a 70B model on a single GPU? (4-bit quantization reduces base model memory by 4×.)

3. You fine-tune on 500 examples for 10 epochs. Training loss is near zero but validation perplexity is worse than before. What happened? (Overfitting.)

4. What is the KL penalty in PPO for? (Prevents the model from diverging too far from the SFT model and losing general capabilities.)

5. DPO does not require a reward model. How does it optimize for human preferences? (It directly optimizes the policy on preference pairs using a binary cross-entropy-like loss.)

---

## 9. Summary

Fine-tuning adapts LLMs to specific tasks. Full fine-tuning is powerful but expensive. PEFT methods (LoRA, QLoRA) achieve similar results with 100× less memory. Instruction fine-tuning teaches task following. RLHF and DPO align models with human preferences. Data quality matters more than quantity. Start with 100-1000 high-quality examples, use LoRA, and evaluate rigorously.

---

## 10. Where to Go Next

- [[RAG]] — An alternative to fine-tuning for knowledge-intensive tasks
- [[Prompt Engineering]] — The simplest form of task adaptation
- [[Training Techniques]] — Advanced training tricks for fine-tuning
