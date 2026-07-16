---
tags: [deep-learning, cnn, advanced]
status: growing
created: 2026-06-27
---

# Redes Neuronales Convolucionales

## 1. Escenario de aprendizaje

Trabajas en una aplicación de diagnóstico médico por imágenes. Tienes radiografías de tórax y necesitas clasificarlas automáticamente para detectar neumonía. Antes de las CNNs (2012), la clasificación de imágenes usaba características diseñadas a mano (bordes, texturas, colores). Después de AlexNet, el paradigma cambió a aprender características visuales directamente de los píxeles. Las CNNs aplican más allá de las imágenes — funcionan con cualquier dato que tenga **estructura espacial o temporal**: audio (1D en el tiempo), texto (1D sobre caracteres), video (3D espacio+tiempo) e incluso grafos (con modificaciones).

---

## 2. La Operación de Convolución

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

### 2.1 Intuición

Una convolución desliza un filtro pequeño (kernel) sobre la entrada, calculando productos punto en cada posición. Cada filtro detecta un patrón específico — bordes, texturas, formas. Ver [[Image Processing Fundamentals]] para más sobre filtros de imagen tradicionales.

**Analogía**: imagina brillar una linterna con un lente con patrón sobre una imagen. En cada posición, el filtro "se ilumina" cuando el patrón coincide.

### 2.2 Cómo Funciona

```
Entrada (5×5):          Kernel (3×3):        Salida (3×3):
[1 1 1 0 0]          [1 0 1]              [4 3 4]
[0 1 1 1 0]          [0 1 0]     →        [2 4 3]
[0 0 1 1 1]          [1 0 1]              [2 3 4]
[0 0 1 1 0]
[0 1 1 0 0]
```

Paso a paso: colocar kernel en la esquina superior izquierda de la entrada → multiplicar elemento por elemento → sumar = 4 → deslizar a la derecha por stride → repetir.

### 2.3 Parámetros Clave

| Parámetro | Efecto |
|---|---|
| **Tamaño del kernel** | Kernels más grandes detectan patrones más grandes pero usan más parámetros. $3\times3$ es el predeterminado — apilar dos $3\times3$ da el mismo receptive field que una $5\times5$ con menos parámetros |
| **Stride** | Tamaño del paso. Stride 2 reduce el tamaño de salida a la mitad. |
| **Padding** | Añadir ceros alrededor de la entrada para preservar tamaño ("same" padding) |
| **Dilatación** | Espacios entre elementos del kernel. Se usa para aumentar el receptive field sin aumentar parámetros |

### 2.4 Por Qué las Convoluciones Son Eficientes

Una capa fully connected que conecta una imagen RGB de $224\times224$ a 1024 neuronas tiene: $224 \times 224 \times 3 \times 1024 \approx 154M$ parámetros.

Una capa convolucional con $64$ kernels de $3\times3$ tiene: $3 \times 3 \times 3 \times 64 = 1,728$ parámetros — **89,000× menos**.

Además, las convoluciones son **invariantes a traslaciones**: un gato es un gato ya sea que aparezca arriba o abajo en la imagen.

---

## 3. Los Bloques de Construcción de CNN

### 3.1 Convolución + Pooling + Activación

El patrón estándar:

```
Entrada → Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → FC → Softmax
```

Cada convolución extrae características. Pooling hace downsampling (reduce tamaño). ReLU añade no-linealidad. Las capas fully connected finales hacen la predicción.

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

## 4. Arquitecturas Clásicas

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

- 2 conv + 3 capas FC
- 60k parámetros
- Resolvió el reconocimiento de dígitos MNIST

### 4.2 AlexNet (2012)

- 5 conv + 3 capas FC
- 60M parámetros
- Ganó ImageNet por un margen enorme (15.3% vs 26.2% de error)
- Introdujo ReLU, dropout, data augmentation, entrenamiento con GPU

### 4.3 VGG (2014)

- 16-19 capas conv, todas $3\times3$
- Arquitectura simple y uniforme
- 138M parámetros (muy grandes)

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

Escala sistemáticamente profundidad, anchura y resolución usando un coeficiente compuesto. Logra precisión de estado del arte con mucho menos parámetros y FLOPs.

---

## 5. Desarrollos Modernos

| Arquitectura | Idea Clave |
|---|---|
| **DenseNet** | Cada capa se conecta a todas las capas siguientes (flujo máximo de información) |
| **MobileNet** | Convoluciones depthwise separables (eficiente para móvil/borde) |
| **ResNeXt** | Convoluciones agrupadas (rutas conv paralelas) |
| **ConvNeXt** | ConvNet modernizado con decisiones de diseño inspiradas en transformers |

---

Para arquitecturas de detección y segmentación basadas en CNNs, ver [[Object Detection & Segmentation]].

## 6. CNNs Más Allá de las Imágenes

- **CNNs 1D**: series temporales, audio, texto (a nivel de caracteres)
- **CNNs 3D**: video (espacio-temporal), volúmenes médicos (CT, MRI)
- **CNNs de Grafos** (GCN): estructuras moleculares, redes sociales, nubes de puntos 3D

---

## 7. Errores Comunes

1. **Tamaño de kernel demasiado grande**: $7\times7$ y más grandes rara vez son necesarios. Apilar $3\times3$ es más eficiente en parámetros y permite más no-linealidad.

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
- [[Training Techniques]] — Optimizadores y regularización para modelos de visión
