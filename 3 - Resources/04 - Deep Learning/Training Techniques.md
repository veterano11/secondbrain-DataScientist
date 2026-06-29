---
tags: [deep-learning, training, advanced]
status: growing
created: 2026-06-27
---

# Training Techniques

## 1. Escenario de aprendizaje

Has diseñado una red neuronal para clasificar imágenes de retina y diagnosticar retinopatía diabética. Conoces la arquitectura (capas convolucionales, fully connected, softmax), pero cuando comienzas a entrenar, la pérdida no disminuye o explota a NaN. Aquí es donde las técnicas de entrenamiento marcan la diferencia. Un optimizador bien elegido, una planificación de learning rate y una estrategia de normalización pueden significar la diferencia entre un modelo que converge en horas y uno que diverge a NaN.

---

## 2. Optimizers

### 2.1 Stochastic Gradient Descent (SGD)

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

The simplest optimizer. Each step moves directly opposite the gradient.

**Pros**: simple, well-understood, good generalization.
**Cons**: slow convergence, sensitive to learning rate, can get stuck in saddle points.

### 2.2 SGD + Momentum

$$v_{t+1} = \beta v_t + \nabla L(w_t)$$
$$w_{t+1} = w_t - \eta v_{t+1}$$

Accumulates past gradients to build velocity. If gradient direction is consistent, momentum accelerates. If it oscillates, momentum smooths it out.

**Intuition**: a ball rolling downhill — it builds momentum in consistent directions and is less affected by local noise.

### 2.3 Adam

$$\text{Adaptive Moment Estimation}$$

The most widely used optimizer. Maintains:
- First moment (mean) of past gradients → momentum
- Second moment (variance) of past gradients → per-parameter learning rate

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Why it is the default**:
- Per-parameter adaptive learning rates (works well without tuning)
- Built-in momentum
- Handles sparse gradients well
- Robust to different problem scales

### 2.4 AdamW

Adam with **decoupled weight decay**. In standard Adam, L2 regularization interacts with the adaptive learning rate. AdamW applies weight decay separately, which is theoretically cleaner and works better for [[Transformers]].

### 2.5 Comparing Optimizers

See [[Hyperparameter Tuning]] for guidance on setting these values.

| Optimizer | Best For | Key Hyperparams |
|---|---|---|
| **SGD** | Image classification, simple tasks | LR, momentum |
| **Adam** | General purpose (default) | LR (3e-4 default), β₁, β₂ |
| **AdamW** | Transformers, LLMs | LR, weight decay |
| **Lion** | Memory-constrained training | LR |

---

## 3. Learning Rate Scheduling

### 3.1 Cosine Decay

$$\eta_t = \frac{1}{2} \eta_0 \left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

Starts at $\eta_0$, smoothly decreases to 0 at step $T$. Encourages the model to explore broadly early and fine-tune late.

### 3.2 Linear Warmup + Cosine Decay

Warmup: linearly increase LR from 0 to $\eta_0$ over the first $W$ steps.
Decay: cosine decay from $\eta_0$ to 0 over remaining steps.

Warmup prevents early instability (large updates when the model is randomly initialized).

### 3.3 Reduce on Plateau

Reduce LR by a factor (e.g., 0.5) when validation loss stops improving for $P$ epochs. Simple and effective.

### 3.4 One Cycle

Warmup to high LR → cosine decay to very low LR. Anneal to very low LR at the end. Known for fast convergence.

---

## 4. Normalization

### 4.1 Batch Normalization

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Normalizes each feature across the batch. Reduces internal covariate shift. Allows higher learning rates. Adds slight regularization (noise from batch statistics).

**Limitation**: batch size must be large enough for reliable statistics. Unstable with batch size 1 or variable-length sequences.

### 4.2 Layer Normalization

Normalizes across features for each sample independently:

$$\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

Independent of batch size. Works for RNNs and Transformers (where batch norm fails due to variable lengths).

### 4.3 RMSNorm

$$y = \frac{x}{\sqrt{\text{RMS}(x) + \epsilon}} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$$

Simplification of LayerNorm — no mean centering. Faster, used in Llama and modern LLMs.

---

## 5. Gradient and Regularization Techniques

See [[Gradient-Based Optimization]] for foundational concepts.

| Technique | What It Does | When To Use |
|---|---|---|
| **Gradient Clipping** | Caps gradient norm to prevent explosion | RNNs, deep transformers, unstable training |
| **Gradient Accumulation** | Sum gradients over multiple batches | Simulating large batch on limited GPU memory |
| **Mixed Precision** | FP16/BF16 with FP32 master weights | 2× faster training, half GPU memory |
| **Label Smoothing** | Soften target labels: $y' = (1-\epsilon)y + \epsilon/K$ | Classification, reduces overconfidence |
| **Stochastic Depth** | Randomly drop layers during training | Very deep networks (1000+ layers) |

---

## 6. Transfer Learning

Training from scratch is rarely optimal. Instead, start from a pre-trained model:

| Strategy | What You Update | Data Needed | Example |
|---|---|---|---|
| **Feature extraction** | Freeze backbone, train classifier | Small | ResNet on custom image dataset |
| **Full fine-tuning** | All parameters | Medium | BERT for sentiment analysis |
| **LoRA** | Low-rank adapters | Small | Llama for custom chat task |
| **Adapter** | Small inserted layers | Small | Any large model |

---

## 7. Distributed Training

For models too large for one GPU:

| Strategy | How It Works |
|---|---|
| **DDP** | Each GPU has full model copy, different data batches, sync gradients |
| **FSDP** | Shard model parameters across GPUs, reconstruct during forward/backward |
| **Tensor Parallel** | Split a single layer's computation across GPUs |
| **Pipeline Parallel** | Put different layers on different GPUs |

FSDP is the most common for training LLMs (splits parameters, gradients, and optimizer states).

---

## 8. Common Mistakes

1. **Learning rate too high**: the most common training failure. If loss oscillates or goes to NaN, reduce LR.

2. **Not using learning rate scheduling**: a constant LR is rarely optimal. Cosine decay or Reduce on Plateau are almost always better.

3. **Batch size too large without adjusting LR**: if you double batch size, double the LR (linear scaling rule). Otherwise, gradients are less stochastic and updates are too small.

4. **Forgetting gradient clipping for RNNs/transformers**: these architectures are prone to gradient explosion. Clip to norm 1.0 as a default.

5. **Not monitoring gradient norms**: watching only the loss can miss instability. Log gradient norms to spot problems early.

---

## 9. Check Your Understanding

1. Adam has two momentum parameters ($\beta_1, \beta_2$). What does each control? ($\beta_1$: gradient direction momentum, $\beta_2$: gradient magnitude momentum)

2. Why does AdamW separate weight decay from the adaptive learning rate? (Standard Adam with L2 regularization scales regularization by the adaptive LR, which is incorrect.)

3. You have a batch size of 32 but can only fit 8 on your GPU. What do you do? (Use gradient accumulation — sum gradients over 4 micro-batches.)

4. LayerNorm normalizes across the feature dimension. BatchNorm normalizes across the batch dimension. When would each be preferred?

5. A 1B parameter model does not fit on one GPU. Which distributed training strategy do you use?

---

## 10. Resumen

Training a neural network effectively requires choosing the right optimizer, learning rate schedule, normalization, and regularization. Adam is the default optimizer. Cosine decay with linear warmup is the default schedule. LayerNorm/RMSNorm are standard for transformers. Gradient clipping prevents explosion. Transfer learning saves data and compute. Pick the right combination and monitor both loss and gradient norms throughout training.

---

## 11. Where to Go Next

- [[Neural Networks]] — What these techniques train
- [[Fine-tuning]] — Applying these techniques to pre-trained models
- [[Regularization]] — Preventing overfitting during training
