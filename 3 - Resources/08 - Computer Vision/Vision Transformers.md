---
tags: [computer-vision, transformers, attention]
status: growing
created: 2026-06-27
---

# Vision Transformers

## 1. Escenario de aprendizaje

Querés clasificar imágenes de razas de perros. Usaste una ResNet-50 y funciona bien. Pero escuchaste que los Vision Transformers (ViT) son el estado del arte. Decidís probar uno. Al hacerlo, te das cuenta de que el modelo no converge en tu dataset de 5000 imágenes. ¿Qué pasó? Resulta que los transformers necesitan 10× más datos que las CNNs.

Esta nota explica por qué, cómo funciona ViT, y cómo elegir entre CNN y Transformer para visión.

## 2. De la CNN al Transformer

Las CNNs tienen **inductive biases** incorporados:
- **Localidad**: un filtro 3×3 solo ve una vecindad chica
- **Translación-equivarianza**: si movés un objeto en la imagen, la CNN lo sigue detectando

Los Transformers no tienen ninguno de estos sesgos. Aprenden relaciones entre píxeles lejanos desde cero. Esto los hace más flexibles, pero necesitan muchos más datos.

## 3. Cómo funciona ViT

### 3.1 Paso a paso

```
Entrada: imagen RGB de 224×224
         ↓
Dividir en parches de 16×16: (224/16)² = 196 parches
         ↓
Aplanar cada parche: 16×16×3 = 768 valores
         ↓
Proyección lineal: cada parche → vector de 768 dimensiones
         ↓
Agregar token [CLS] → 197 vectores
         ↓
Agregar positional embedding (aprender posición de cada parche)
         ↓
12–24 capas de Transformer encoder (self-attention + FFN)
         ↓
El token [CLS] de la salida → clasificación final
```

```python
import torch
import torch.nn as nn

class SimplifiedViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768, num_layers=12, num_classes=1000):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2  # 196
        patch_dim = patch_size * patch_size * 3      # 768

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=12),
            num_layers=num_layers
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # Dividir en parches y aplanar
        x = x.reshape(B, 3, 224//16, 16, 224//16, 16)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, 196, 768)
        # Proyección lineal
        x = self.patch_embed(x)                       # (B, 196, 768)
        # Agregar token [CLS]
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)               # (B, 197, 768)
        # Agregar positional embedding
        x = x + self.pos_embed                       # (B, 197, 768)
        # Transformer encoder
        x = self.encoder(x)                          # (B, 197, 768)
        # Clasificar desde [CLS]
        return self.head(x[:, 0])                    # (B, num_classes)
```

### 3.2 Atención

Self-attention en ViT funciona igual que en NLP:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Cada parche se relaciona con todos los otros parches. Un parche de ojo de perro puede atender al parche de oreja a 20 parches de distancia — algo que una CNN requeriría muchas capas para lograr.

### 3.3 Costo computacional

| Modelo | Parches | Tokens | Costo atención |
|--------|---------|--------|----------------|
| ViT-B/16 | 16×16 | 197 | $197^2 \times 12$ cabezas ≈ 460K |
| ViT-L/16 | 16×16 | 197 | $197^2 \times 16$ ≈ 620K |
| ViT-B/8 | 8×8 | 28²+1 = 785 | $785^2 \times 12$ ≈ 7.4M (16× más caro) |

**Parche más chico = más tokens = atención O(n²) mucho más cara**. Por eso ViT suele usar parches de 16×16 o 14×14.

## 4. Datos: el punto débil de ViT

ViT entrenado desde cero en ImageNet (1.3M imágenes): **peor que ResNet-50**.

ViT pre-entrenado en JFT-300M (300M imágenes): **mejor que cualquier CNN**.

| Dataset | Tamaño | ViT vs CNN |
|---------|--------|------------|
| ImageNet-1K | 1.3M | CNN gana |
| ImageNet-21K | 14M | Empate |
| JFT-300M | 300M | ViT gana |

**Moraleja**: si tenés < 10M imágenes, no uses ViT desde cero. Usá CNN o DeiT (data-efficient ViT con distillación de una CNN maestra).

## 5. Arquitecturas que mejoran ViT

| Modelo | Problema que resuelve | Cómo lo hace |
|--------|----------------------|--------------|
| **DeiT** | ViT necesita muchos datos | Distillación de una CNN como teacher + data augmentation fuerte |
| **Swin** | ViT no tiene jerarquía (una sola escala) | Ventanas desplazadas + diseño piramidal como CNN |
| **CvT** | Parches fijos pierden info local | Embedding convolucional + depthwise conv en atención |
| **MaxViT** | Atención global es cara en alta resolución | Atención multi-eje: local bloqueada + global diluida |
| **DETR** | Detección con anchors es compleja | Transformer encoder-decoder + Hungarian matching, sin NMS |

### Swin Transformer en detalle

En lugar de atención global, Swin divide el feature map en ventanas no-superpuestas y aplica atención dentro de cada ventana. Entre capas, desplaza las ventanas para permitir comunicación cruzada.

```
Capa 1: atención en ventanas 7×7, sin solapamiento
Capa 2: ventanas desplazadas → conexiones entre ventanas previas
```

Esto da jerarquía espacial 4× → 8× → 16× → 32×, como una CNN. Swin es el backbone de arquitecturas como Mask R-CNN y BEiT.

## 6. ViT vs CNN: cuándo usar cuál

| Situación | Recomendación |
|-----------|--------------|
| Dataset < 1M imágenes | CNN o DeiT |
| Dataset > 10M imágenes | ViT |
| Presupuesto de cómputo ajustado | CNN (más eficiente) |
| Memoria de GPU limitada | Swin o CNN |
| Necesitás features multi-escala | Swin (piramidal) |
| Trabajás con video (muchos frames) | ViT + factorización temporal |

## 7. Common Mistakes

1. **ViT sin suficiente aumento de datos**: recortes aleatorios, color jitter, mixup, cutmix — ViT necesita todo. Sin esto, overfitea feo.
2. **Positional encoding que no generaliza**: si entrenaste con resolución 224 y querés usar 448, los embeddings aprendidos no funcionan. Necesitás interpolación.
3. **Usar ViT para segmentación sin arquitectura piramidal**: ViT da una sola escala. Para segmentación necesitás algo como Swin o ViT-Adapter.
4. **Confundir ViT con MLP-Mixer**: ViT tiene self-attention. MLP-Mixer usa MLPs en lugar de atención. No son lo mismo.

## 8. Check Your Understanding

1. ViT sin pre-training en JFT-300M rinde peor que ResNet-50 en ImageNet. ¿Por qué exactamente?
2. Si Swin usa ventanas chicas, ¿cómo logra comunicación global?
3. DETR no usa anchors ni NMS. ¿Cómo resuelve la correspondencia entre predicciones y ground truth?
4. Una imagen de 448×448 con parches de 16×16 genera cuántos tokens? ¿Cuánto más caro es que 224×224?

**Respuestas rápidas:**
1. ViT no tiene inductive bias de localidad; necesita ver millones de ejemplos para aprender que píxeles cercanos se relacionan.
2. Las ventanas se desplazan entre capas → con 2 capas, cada token ve indirectamente toda la imagen.
3. Hungarian matching: asigna cada predicción al ground truth más cercano, optimizando la asignación global.
4. (448/16)² = 784 tokens. Atención O(n²): (784/197)² ≈ 16× más caro que 224².

## 9. Summary

Vision Transformer adapta el Transformer de NLP a imágenes dividiéndolas en parches y tratándolos como tokens de secuencia. Es más flexible que las CNNs (no tiene sesgos de localidad prefijados), pero necesita 10-100× más datos para alcanzar su potencial. Arquitecturas como Swin y DeiT resuelven limitaciones específicas (jerarquía y datos insuficientes). La elección entre ViT y CNN depende del tamaño del dataset y los recursos de cómputo.

## 10. Where to Go Next

- [[Transformers]] — la arquitectura original que ViT adapta
- [[Object Detection & Segmentation]] — DETR y Mask2Former son transformer-based
- [[Self-Supervised & Multimodal Vision]] — DINO, MAE, CLIP pre-entrenan ViTs sin etiquetas
- [[CNNs]] — la arquitectura que ViT compite y complementa
- [[Image Processing Fundamentals]] — cómo se preparan las imágenes antes de entrar a ViT
- [[Transfer Learning]] — pre-entrenamiento de ViT en grandes datasets
- [[Training Techniques]] — aumento de datos necesario para ViT
