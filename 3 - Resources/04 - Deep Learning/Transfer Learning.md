---
tags: [deep-learning, transfer-learning, advanced]
status: growing
created: 2026-06-27
---

# Transfer Learning

## 1. Escenario de aprendizaje

Tienes 500 imágenes de rayos X para clasificar entre neumonía y normal. Entrenar una red neuronal profunda desde cero requeriría millones de imágenes etiquetadas, días de GPU y experiencia significativa. Transfer learning — comenzar desde un modelo ya entrenado en un gran conjunto de datos genérico (como ImageNet) — reduce esto a cientos de ejemplos y horas de fine-tuning en una sola GPU. Prácticamente toda aplicación práctica de deep learning usa transfer learning. No es una técnica especializada; es la práctica estándar.

---

## 2. The Core Idea

**Pre-training**: train a model on a large, generic dataset (ImageNet for vision, Wikipedia + books for NLP).

**Fine-tuning**: adapt the pre-trained model to your specific task with a smaller dataset.

**Why it works**: early layers learn general features (edges, textures in vision; syntax, grammar in language). Only the last few layers need to specialize to the target task.

---

## 3. Transfer Learning Strategies

### 3.1 Feature Extraction

- Freeze the pre-trained backbone (all weights stay fixed)
- Replace the final classification layer(s)
- Train only the new layers

**When to use**: very small target dataset (< 1000 examples), similar domain to pre-training data.

```python
import torchvision.models as models

# Load pre-trained ResNet
backbone = models.resnet50(weights="IMAGENET1K_V2")

# Freeze all layers
for param in backbone.parameters():
    param.requires_grad = False

# Replace classifier
backbone.fc = nn.Linear(2048, num_classes)

# Only the new classifier layer is trainable
```

### 3.2 Full Fine-Tuning

- Unfreeze the entire model
- Train everything with a low learning rate

**When to use**: moderate target dataset (1000-10000 examples), or different domain from pre-training.

### 3.3 Progressive Unfreezing

- Start with only the new classifier trainable
- Gradually unfreeze layers from top to bottom
- Each unfreeze step uses a lower learning rate

**Why**: early layers learn very general features that should change the least. Gradually exposing them to the target data prevents catastrophic forgetting.

### 3.4 Parameter-Efficient Fine-Tuning

Instead of updating all parameters, insert small trainable modules:

| Method | What is trained | Parameters | Used in |
|---|---|---|---|
| **LoRA** | Low-rank weight updates | ~0.1-1% | LLMs |
| **Adapters** | Small bottleneck layers | ~1-5% | NLP |
| **Prefix Tuning** | Virtual token embeddings | ~0.1% | NLG |

---

## 4. Common Pre-trained Models

### 4.1 Vision

| Model | Pre-training Data | Best For |
|---|---|---|
| **ResNet** (18/50/101) | ImageNet (1.3M images) | Classification, detection — see [[CNNs]] |
| **EfficientNet** | ImageNet + noisy student | Efficiency |
| **ViT** (Vision Transformer) | ImageNet-21k / JFT-300M | Vision (transformer-based) — see [[Transformers]] |
| **CLIP** (OpenAI) | 400M image-text pairs | Zero-shot, multi-modal |

### 4.2 NLP / LLMs

| Model | Parameters | Pre-training Data | Best For |
|---|---|---|---|
| **BERT** | 110M-340M | Books + Wikipedia | Understanding (classification, NER) |
| **RoBERTa** | 125M-355M | 160GB text | Understanding (improved BERT) |
| **GPT-2** | 124M-1.5B | WebText | Generation |
| **Llama 2/3** | 7B-70B | 2T-15T tokens | General purpose |
| **Mistral** | 7B | Various | General purpose (efficient) |
| **Gemma** | 2B-7B | 6T tokens | Research |

---

## 5. Practical Guidelines

### 5.1 Choosing a Strategy

| Target Data | Domain Similarity | Recommended Strategy |
|---|---|---|
| < 1000 | Similar | Feature extraction |
| < 1000 | Different | Fine-tune top layers |
| 1K-10K | Similar | Fine-tune full model with low LR |
| 1K-10K | Different | Full fine-tune from pre-trained |
| > 10K | Any | Consider training from scratch |

### 5.2 Fine-Tuning Best Practices

- **Learning rate**: 10-100× lower than training from scratch. For full fine-tuning: 2e-5 to 5e-5. For LoRA: 1e-4 to 5e-4.
- **Optimizer**: AdamW is standard. SGD with momentum works for vision.
- **Epochs**: fewer than training from scratch. Monitor validation loss — fine-tuning can overfit quickly.
- **Batch size**: as large as memory allows (but not too large — small batches act as regularizer).
- **Data augmentation**: even more important with small datasets.

### 5.3 When NOT to Transfer Learn

- Target domain is fundamentally different from pre-training domain (e.g., medical images from a novel imaging modality)
- You have a very large target dataset (millions of examples)
- Latency constraints require a much smaller model
- You want to understand the architecture from the ground up (research)

---

## 6. Common Mistakes

1. **Learning rate too high**: the most common fine-tuning mistake. Pre-trained weights are already good — large updates destroy learned features.

2. **Not freezing batch norm statistics**: batchnorm running mean/var should be frozen when fine-tuning with small batches. Update them only if training with large batches.

3. **Overfitting to small data**: fine-tuning on hundreds of examples can still overfit. Use stronger [[Regularization]], early stopping, and data augmentation.

4. **Catastrophic forgetting**: the model may "forget" the general features learned during pre-training. Use progressive unfreezing or replay of pre-training data.

5. **Not adjusting the input size**: pre-trained models expect specific input sizes (224×224 for ResNet, 512/1024 tokens for BERT). Resize your data accordingly.

---

## 7. Check Your Understanding

1. You have 500 labeled medical X-ray images. Should you train ResNet from scratch, use feature extraction, or fine-tune the full model? Why?

2. Why is the learning rate for fine-tuning typically 100× lower than training from scratch?

3. LoRA trains 1% of parameters but achieves similar performance to full fine-tuning on LLMs. How?

4. You fine-tune a BERT model on a custom dataset. Training loss decreases but validation loss increases after 2 epochs. What do you do?

5. When would you NOT use transfer learning?

---

## 8. Summary

Transfer learning is the standard practice in deep learning. Start from a pre-trained model, adapt it to your task with a small amount of data and a low learning rate. Choose the strategy (feature extraction, fine-tuning, or PEFT) based on your dataset size and domain similarity. The key is not to destroy the pre-trained features with aggressive updates.

---

## 9. Where to Go Next

- [[Neural Networks]] — What is being transferred
- [[Fine-tuning]] — LLM-specific transfer learning (LoRA, QLoRA)
- [[Training Techniques]] — Optimizers and schedules for fine-tuning
