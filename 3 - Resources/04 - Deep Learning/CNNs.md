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

**Max pooling**: take the maximum value in each window. Preserves strongest features, discards spatial detail.

**Average pooling**: take the average. Preserves overall activation level.

**Global average pooling**: average the entire feature map. Often used before the final layer to avoid overfitting (no parameters to learn).

---

## 4. Classic Architectures

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

**The breakthrough**: residual connections.

$$y = F(x) + x$$

Instead of learning $F(x)$ directly, the network learns the **residual** $F(x) = H(x) - x$. If the optimal mapping is the identity, the network can learn $F(x) = 0$ (easy) instead of $H(x) = x$ (hard).

**Why this matters**: it allowed training 152-layer networks (previous best was ~20). Residual connections ensure gradients can flow directly to early layers.

ResNet variants (ResNet-50, ResNet-101, ResNet-152) remain popular as feature extractors.

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

## 7. Common Mistakes

1. **Kernel size too large**: $7\times7$ and larger are rarely needed. Stack of $3\times3$ is more parameter-efficient and allows more non-linearity.

2. **Too much pooling too fast**: aggressive downsampling loses spatial information. Pool gradually.

3. **Forgetting that CNNs expect fixed-size inputs**: fully connected layers require fixed input size. Use global average pooling or adaptive pooling to handle variable sizes.

4. **Not using data augmentation**: vision models benefit enormously from augmentation (rotation, flip, color jitter). Without it, they overfit badly. See [[Regularization]] for more augmentation techniques.

5. **Pretrained vs scratch**: unless you have millions of images, use a pretrained model (transfer learning). Training from scratch is rarely justified.

---

## 8. Check Your Understanding

1. A $3\times3$ convolution on a $224\times224\times3$ input with 64 filters, stride 1, same padding. What is the output shape? How many parameters?

2. Why does stacking two $3\times3$ convolutions give the same receptive field as one $5\times5$? Which is better and why?

3. ResNet uses skip connections. How do they help with vanishing gradients?

4. You have 1000 images of dogs and cats. Should you train a CNN from scratch or use transfer learning? Why?

5. What is the difference between max pooling and average pooling? When would you use each?

---

## 9. Summary

CNNs use convolutions to efficiently learn spatial patterns. The convolution operation detects local features, pooling downsamples, and the composition of many layers builds hierarchical representations (edges → textures → parts → objects). Key innovations — residual connections, depthwise convolutions, and compound scaling — have pushed accuracy while reducing parameters.

---

## 10. Where to Go Next

- [[Neural Networks]] — Foundational concepts for CNNs
- [[Transfer Learning]] — Using pretrained CNNs for new tasks
- [[Training Techniques]] — Optimizers and regularization for vision models
