---
tags: [deep-learning, cnn, advanced]
status: growing
created: 2026-06-27
---

# Convolutional Neural Networks

## 1. Escenario de aprendizaje

Trabajas en una aplicación de diagnóstico médico por imágenes. Tienes radiografías de tórax y necesitas clasificarlas automáticamente para detectar neumonía. Antes de las CNNs (2012), la clasificación de imágenes usaba características diseñadas a mano (bordes, texturas, colores). Después de AlexNet, el paradigma cambió a aprender características visuales directamente de los píxeles. Las CNNs aplican más allá de las imágenes — funcionan con cualquier dato que tenga **estructura espacial o temporal**: audio (1D en el tiempo), texto (1D sobre caracteres), video (3D espacio+tiempo) e incluso grafos (con modificaciones).

---

## 2. The Convolution Operation

### Arquitectura Visual de CNN

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA CNN                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT        CONV+ReLU     POOL        CONV+ReLU     POOL        │
│  ┌─────┐     ┌─────┐      ┌─────┐     ┌─────┐      ┌─────┐       │
│  │     │────▶│     │─────▶│     │────▶│     │─────▶│     │       │
│  │     │     │     │      │     │     │     │      │     │       │
│  └─────┘     └─────┘      └─────┘     └─────┘      └─────┘       │
│  32×32×3      32×32×64     16×16×64    16×16×128    8×8×128       │
│                                                                     │
│  CAPA FINAL: Flatten → Dense → Softmax → PREDICCIÓN               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Intuition

A convolution slides a small filter (kernel) over the input, computing dot products at each position. Each filter detects a specific pattern — edges, textures, shapes. See [[Image Processing Fundamentals]] for more on traditional image filters.

**Analogy**: imagine shining a flashlight with a patterned lens over an image. At each position, the filter "lights up" when the pattern matches.

### 2.2 How It Works

```
Input (5×5):          Kernel (3×3):        Output (3×3):
[1 1 1 0 0]          [1 0 1]              [4 3 4]
[0 1 1 1 0]          [0 1 0]     →        [2 4 3]
[0 0 1 1 1]          [1 0 1]              [2 3 4]
[0 0 1 1 0]
[0 1 1 0 0]
```

Step by step: place kernel at top-left of input → multiply element-wise → sum = 4 → slide right by stride → repeat.

### 2.3 Key Parameters

| Parameter | Effect |
|---|---|
| **Kernel size** | Larger kernels detect larger patterns but use more parameters. $3\times3$ is the default — stacking two $3\times3$ gives same receptive field as one $5\times5$ with fewer parameters |
| **Stride** | Step size. Stride 2 reduces output size by half. |
| **Padding** | Add zeros around input to preserve size ("same" padding) |
| **Dilation** | Gaps between kernel elements. Used to increase receptive field without increasing parameters |

### 2.4 Why Convolutions Are Efficient

A fully connected layer connecting a $224\times224$ RGB image to 1024 neurons has: $224 \times 224 \times 3 \times 1024 \approx 154M$ parameters.

A convolutional layer with $64$ kernels of $3\times3$ has: $3 \times 3 \times 3 \times 64 = 1,728$ parameters — **89,000× fewer**.

Plus, convolutions are **translation invariant**: a cat is a cat whether it appears at the top or bottom of the image.

---

## 3. The CNN Building Blocks

### 3.1 Convolution + Pooling + Activation

The standard pattern:

```
Input → Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → FC → Softmax
```

Each convolution extracts features. Pooling downsamples (reduces size). ReLU adds non-linearity. The final fully connected layers make the prediction.

### 3.2 Pooling

#### Comparación de Métodos de Pooling

```
MAX POOLING (2×2, stride 2):

┌─────┬─────┬─────┬─────┐      ┌─────┬─────┐
│  1  │  3  │  2  │  1  │      │  4  │  3  │
├─────┼─────┼─────┼─────┤  ──▶ ├─────┼─────┤
│  4  │  2  │  3  │  1  │      │  5  │  4  │
├─────┼─────┼─────┼─────┤      └─────┴─────┘
│  5  │  1  │  4  │  2  │
├─────┼─────┼─────┼─────┤
│  3  │  2  │  1  │  3  │
└─────┴─────┴─────┴─────┘

AVERAGE POOLING (2×2, stride 2):

┌─────┬─────┬─────┬─────┐      ┌─────┬─────┐
│  1  │  3  │  2  │  1  │      │ 2.5 │ 1.5 │
├─────┼─────┼─────┼─────┤  ──▶ ├─────┼─────┤
│  4  │  2  │  3  │  1  │      │ 2.5 │ 2.5 │
├─────┼─────┼─────┼─────┤      └─────┴─────┘
│  5  │  1  │  4  │  2  │
├─────┼─────┼─────┼─────┤
│  3  │  2  │  1  │  3  │
└─────┴─────┴─────┴─────┘
```

---

## 4. Classic Architectures

### Comparación de Arquitecturas CNN

```
┌─────────────────────────────────────────────────────────────────────┐
│              EVOLUCIÓN DE ARQUITECTURAS CNN                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LeNet (1998)    AlexNet (2012)    VGG (2014)    ResNet (2015)     │
│  ┌─────┐        ┌─────┐          ┌─────┐        ┌─────┐           │
│  │ 2   │        │ 5   │          │ 16  │        │ 152 │           │
│  │capas│        │capas│          │capas│        │capas│           │
│  └─────┘        └─────┘          └─────┘        └─────┘           │
│  60K params     60M params       138M params    25M params        │
│                                                                     │
│  MNIST          ImageNet #1      ImageNet       ImageNet #1        │
│                 (top-5: 15.3%)  (top-5: 7.3%) (top-5: 3.6%)      │
│                                                                     │
│  CLAVE:           CLAVE:           CLAVE:          CLAVE:           │
│  Primera CNN      ReLU, Dropout    Uniformidad     Skip Connections │
│  funcional        Data Augment.    $3\times3$      (Residual)       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.1 LeNet-5 (1998)

- 2 conv + 3 FC layers
- 60k parameters
- Solved MNIST digit recognition

### 4.2 AlexNet (2012)

- 5 conv + 3 FC layers
- 60M parameters
- Won ImageNet by a huge margin (15.3% vs 26.2% error)
- Introduced ReLU, dropout, data augmentation, GPU training

### 4.3 VGG (2014)

- 16-19 conv layers, all $3\times3$
- Simple, uniform architecture
- 138M parameters (very large)

### 4.4 ResNet (2015)

**El avance**: conexiones residuales.

$$y = F(x) + x$$

En lugar de aprender $F(x)$ directamente, la red aprende el **residual** $F(x) = H(x) - x$. Si el mapping óptimo es la identidad, la red puede aprender $F(x) = 0$ (fácil) en lugar de $H(x) = x$ (difícil).

**Por qué importa**: permitió entrenar redes de 152 capas (el anterior mejor era ~20). Las conexiones residuales aseguran que los gradientes puedan fluir directamente a las capas tempranas.

```
CONEXIÓN RESIDUAL:
                    
    ┌───────────────┐
    │               │
    ▼               │
┌─────┐        ┌─────┐
│Conv │───────▶│ +   │───────▶ Salida
└─────┘        └─────┘
    │               ▲
    │   ┌─────┐     │
    └──▶│Conv │─────┘
        └─────┘
        
    x ─────────────▶ +
    
    y = F(x) + x
```

Las variantes ResNet (ResNet-50, ResNet-101, ResNet-152) siguen populares como extractores de características.

### 4.5 EfficientNet (2019)

Systematically scales depth, width, and resolution using a compound coefficient. Achieves state-of-the-art accuracy with much fewer parameters and FLOPs.

---

## 5. Modern Developments

| Architecture | Key Idea |
|---|---|
| **DenseNet** | Each layer connects to all subsequent layers (maximum information flow) |
| **MobileNet** | Depthwise separable convolutions (efficient for mobile/edge) |
| **ResNeXt** | Grouped convolutions (parallel conv paths) |
| **ConvNeXt** | Modernized ConvNet with transformer-inspired design choices |

---

For detection and segmentation architectures built on CNNs, see [[Object Detection & Segmentation]].

## 6. CNNs Beyond Images

- **1D CNNs**: time series, audio, text (character-level)
- **3D CNNs**: video (spatio-temporal), medical volumes (CT, MRI)
- **Graph CNNs** (GCN): molecular structures, social networks, 3D point clouds

---

## 7. Errores Comunes

1. **Kernel size demasiado grande**: $7\times7$ y más grandes rara vez son necesarios. Apilar $3\times3$ es más eficiente en parámetros y permite más no-linealidad.

2. **Demasiado pooling demasiado rápido**: el downsampling agresivo pierde información espacial. Pool gradualmente.

3. **Olvidar que las CNNs esperan entradas de tamaño fijo**: las capas fully connected requieren tamaño de entrada fijo. Usa global average pooling o adaptive pooling para manejar tamaños variables.

4. **No usar data augmentation**: los modelos de visión se benefician enormemente de augmentation (rotación, flip, color jitter). Sin él, hacen overfitting severamente. Ver [[Regularization]] para más técnicas de augmentation.

5. **Pretrained vs desde cero**: a menos que tengas millones de imágenes, usa un modelo preentrenado (transfer learning). Entrenar desde cero rara vez está justificado.

---

## 8. Comprueba tu Conocimiento

1. Una convolución $3\times3$ en una entrada $224\times224\times3$ con 64 filtros, stride 1, same padding. ¿Cuál es la forma de salida? ¿Cuántos parámetros tiene?

2. ¿Por qué apilar dos convoluciones $3\times3$ da el mismo receptive field que una $5\times5$? ¿Cuál es mejor y por qué?

3. ResNet usa skip connections. ¿Cómo ayudan con el problema de gradientes que desaparecen?

4. Tienes 1000 imágenes de perros y gatos. ¿Debes entrenar una CNN desde cero o usar transfer learning? ¿Por qué?

5. ¿Cuál es la diferencia entre max pooling y average pooling? ¿Cuándo usarías cada uno?

---

## 9. Resumen

Las CNNs usan convoluciones para aprender patrones espaciales de forma eficiente. La operación de convolución detecta características locales, el pooling hace downsampling, y la composición de muchas capas construye representaciones jerárquicas (bordes → texturas → partes → objetos). Las innovaciones clave — conexiones residuales, convoluciones depthwise y escalado compuesto — han empujado la precisión mientras reducen parámetros.

---

## 10. ¿Dónde ir Siguente?

- [[Neural Networks]] — Conceptos fundacionales para CNNs
- [[Transfer Learning]] — Usar CNNs preentrenadas para nuevas tareas
- [[Object Detection & Segmentation]] — Arquitecturas de detección y segmentación basadas en CNNs
- [[Training Techniques]] — Optimizers and regularization for vision models
