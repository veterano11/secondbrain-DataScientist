---
tags: [computer-vision, self-supervised, multimodal, contrastive]
status: growing
created: 2026-06-27
---

# Self-Supervised & Multimodal Vision

## 1. Escenario de aprendizaje

Tenés 1 millón de imágenes sin etiquetar y solo 1000 etiquetadas. Entrenar un clasificador desde cero con 1000 imágenes da resultados pobres. Pero con self-supervised learning, podés usar el millón de imágenes sin etiquetar para aprender representaciones visuales ricas, y luego fine-tunear con las 1000 etiquetadas para obtener resultados casi tan buenos como si tuvieras 100K etiquetas.

Self-supervised learning (SSL) es el puente entre datos no etiquetados (abundantes) y modelos de alto rendimiento.

## 2. La idea central

Diseñamos una "tarea pretexto": el modelo aprende representaciones resolviendo una tarea para la que las etiquetas vienen gratis.

**Ejemplos de tareas pretexto:**
- Predecir la rotación de una imagen (Rotation Prediction)
- Completar parches enmascarados (Masked Autoencoder)
- Maximizar similitud entre dos vistas aumentadas de la misma imagen (Contrastive Learning)

## 3. Contrastive Learning (SimCLR)

La idea más influyente. Para cada imagen, creamos dos vistas con aumentos aleatorios (crop, color jitter, blur). El modelo aprende a **acercar** embeddings de la misma imagen y **alejar** embeddings de imágenes diferentes.

### 3.1 NT-Xent loss

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

```python
def nt_xent_loss(z1, z2, temperature=0.5):
    # z1, z2: (B, D) — embeddings de dos vistas
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)   # (2B, D)
    z = F.normalize(z, dim=1)

    # Matriz de similitud coseno
    sim = z @ z.T / temperature       # (2B, 2B)

    # Cada imagen tiene su positivo en posición (i, i+B) y (i+B, i)
    labels = torch.arange(B).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss
```

### 3.2 Receta de SimCLR

| Ingrediente | Valor | Efecto |
|-------------|-------|--------|
| Batch size | 4096+ | Suficientes negativos |
| Augmentations | Crop + color jitter + blur + grayscale | Suficiente variación |
| Projection head | MLP (dim → 2048 → 128) | Mejora representaciones |
| Temperature τ | 0.5 | Controla "suavidad" |

## 4. MoCo (Momentum Contrast)

SimCLR necesita batch enorme. MoCo mantiene una **cola** de negativos de batches anteriores + un **momentum encoder** para consistencia:

```text
Batch actual (query): q = f_q(x_q)
Cola de negativos: [k_1, k_2, ..., k_65536]
Momentum encoder: f_k(x) — actualizado como EMA de f_q

Loss: q·k_pos / τ vs q·k_neg / τ para todos los negativos en cola
```

**Ventaja**: batch size independiente del número de negativos. Funciona con batch 256.

## 5. DINO (Self-Distillation with No Labels)

No usa pares positivos/negativos. Un **teacher** (momentum) genera targets, un **student** aprende a predecirlos.

```python
# Conceptual DINO
teacher = vit()  # momentum encoder (sin gradientes)
student = vit()  # entrenado

for x in unlabeled_loader:
    x1, x2 = augment(x), augment(x)  # dos vistas

    t_out = teacher(x1).detach()           # target
    s_out = student(x2)                     # predicción

    loss = cross_entropy(s_out, t_out)
    loss.backward()
    update(student)
    teacher = momentum_update(teacher, student)  # EMA
```

**Propiedad emergente**: los mapas de atención de DINO segmentan objetos sin supervisión — la atención del token [CLS] resalta la silueta del objeto principal.

## 6. CLIP (Contrastive Language-Image Pre-training)

Aprende un espacio conjunto imagen-texto usando 400M pares (imagen, texto) de internet.

```python
# Batch de (imágenes, textos)
I = image_encoder(images)     # (B, 256)
T = text_encoder(texts)       # (B, 256)

# Normalizar
I = F.normalize(I, dim=1)
T = F.normalize(T, dim=1)

# Matriz de similitud (B×B)
logits = I @ T.T * np.exp(temperature)
labels = torch.arange(B)  # diagonal = pares correctos

loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
```

**Zero-shot classification**: sin fine-tuning, clasifica nuevas categorías:

```python
text_prompts = ["a photo of a dog", "a photo of a cat", ...]
text_embeds = text_encoder(text_prompts)

image_embed = image_encoder(image)
probs = softmax(image_embed @ text_embeds.T)
```

**Limitaciones**: CLIP falla en atributos (color, tamaño, conteo) y conceptos abstractos.

## 7. MAE (Masked Autoencoders)

Enmascara 75% de los parches de la imagen y reconstruye los píxeles faltantes. Simétrico a BERT pero para visión.

```python
# MAE conceptual
parches = patchify(image)       # (N, 256)
mascara = random_mask(0.75)    # 25% visibles

visible = parches[~mascara]      # solo el 25% no enmascarado
encoded = encoder(visible)       # transformer solo en visible

# Agregar tokens de máscara y decodificar
tokens = add_mask_tokens(encoded, mascara)
decoded = decoder(tokens)        # decoder ligero

loss = MSE(decoded[mascara], parches[mascara])  # solo en píxeles enmascarados
```

**Por qué funciona**: enmascarar el 75% fuerza al modelo a entender semántica de alto nivel, no solo textura local.

## 8. Modelos más allá de CLIP

| Modelo | Idea |
|--------|------|
| **BLIP-2** | Q-Formatter conecta imagen encoder + LLM congelados |
| **LLaVA** | Proyección lineal simple de CLIP → Vicuna |
| **Flamingo** | Interleaved images-text con gating |
| **ImageBind** | Embedding conjunto de 6 modalidades (imagen, texto, audio, etc.) |

## 9. Common Mistakes

1. **Batch size chico en SimCLR**: < 512 → no hay suficientes negativos. Usá MoCo o incrementá batch.
2. **Augmentaciones débiles**: sin color jitter, el modelo aprende a coincidir basado en color en vez de contenido.
3. **CLIP zero-shot no es mágico**: funciona bien para categorías generales ("perro", "gato") pero mal para específicas ("golden retriever con juguete rojo").
4. **MAE decoder muy pesado**: si el decoder es tan grande como el encoder, perdés la eficiencia del diseño asimétrico.

## 10. Check Your Understanding

1. Contrastive learning necesita pares positivos y negativos. ¿Por qué no puede usar solo positivos? (Sin negativos, todas las imágenes colapsan al mismo embedding)
2. ¿Por qué DINO no necesita pares negativos? (La combinación de centering + sharpening previene el colapso)
3. ¿Cuándo usarías MAE en vez de SimCLR? (Cuando tenés imágenes de alta resolución y querés reconstruir detalles finos)

## 11. Summary

Self-supervised learning aprende representaciones sin etiquetas. Contrastive learning (SimCLR, MoCo) maximiza similitud entre vistas de la misma imagen. DINO usa self-distillation y produce atención segmentada. CLIP aprende un espacio imagen-texto conjunto para zero-shot classification. MAE reconstruye parches enmascarados. SSL es la técnica estándar para pre-entrenar modelos visuales cuando las etiquetas son escasas.

## 12. Where to Go Next

- [[Transfer Learning]] — cómo fine-tunear modelos pre-entrenados con SSL
- [[Vision Transformers]] — ViT es el backbone de DINO, MAE, y CLIP
- [[Generative Models for Vision]] — CLIP guidance para texto→imagen
- [[RAG]] — multimodal retrieval con embeddings CLIP
- [[CNNs]] — backbone alternativo a ViT para SSL
- [[Unsupervised Learning]] — SSL como subcampo del aprendizaje no supervisado
- [[Training Techniques]] — aumentos y estrategias de entrenamiento contrastivo
