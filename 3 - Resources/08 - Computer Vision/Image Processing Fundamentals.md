---
tags: [computer-vision, image-processing, fundamentals]
status: growing
created: 2026-06-27
---

# Image Processing Fundamentals

## 1. Escenario de aprendizaje

Tenés una foto de un documento escaneado: está oscura, tiene bordes borrosos, y querés extraer automáticamente las regiones de texto. Antes de meterla a una red neuronal, el procesamiento clásico de imágenes resuelve el 80% del problema: corregir iluminación, detectar bordes, segmentar regiones.

Al terminar esta nota, vas a poder cargar una imagen, aplicar filtros, detectar bordes, y segmentar regiones — todo con código que corre en segundos, sin GPU.

## 2. Requisitos

```bash
pip install numpy scipy pillow scikit-image matplotlib
```

## 3. La imagen como dato numérico

Una imagen en escala de grises es una matriz 2D donde cada celda es un píxel con intensidad entre 0 (negro) y 255 (blanco). Una imagen a color son tres matrices apiladas: Rojo, Verde, Azul (RGB).

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Cargar imagen y ver su estructura
img = np.array(Image.open('documento.jpg').convert('L'))  # a grises
print(f"Dimensiones: {img.shape}")     # (1024, 768)
print(f"Tipo de dato: {img.dtype}")     # uint8
print(f"Valor mínimo: {img.min()}, máximo: {img.max()}")

# Visualizar
plt.imshow(img, cmap='gray')
plt.title(f"Imagen original — {img.shape}")
plt.show()
```

**Salida esperada:**

```
Dimensiones: (1024, 768)
Tipo de dato: uint8
Valor mínimo: 23, máximo: 211
```

Cada píxel es un número. Procesar una imagen es manipular estos números.

## 4. Convolución y filtrado

Un kernel (o filtro) es una matriz chica que se desliza sobre la imagen. En cada posición, se multiplica elemento a elemento con los píxeles que cubre y se suma. El resultado es un nuevo píxel.

$$(I * K)(x, y) = \sum_{i=-1}^{1} \sum_{j=-1}^{1} I(x+i, y+j) \cdot K(i, j)$$

### 4.1 Suavizado (Gaussian Blur)

Un filtro Gaussiano promedia píxeles vecinos con pesos mayores en el centro. Útil para reducir ruido antes de detectar bordes.

```python
from scipy.ndimage import gaussian_filter

img_suave = gaussian_filter(img.astype(float), sigma=2.0)

print(f"Antes: std={img.std():.1f}")
print(f"Después: std={img_suave.std():.1f}")  # menor → imagen más pareja
```

**Salida esperada:**
```
Antes: std=45.2
Después: std=38.7
```

### 4.2 Detección de bordes con Sobel

El filtro Sobel calcula el gradiente de la imagen — regiones de cambio brusco de intensidad son bordes.

```python
from scipy.ndimage import convolve

# Kernel Sobel para bordes verticales
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Kernel Sobel para bordes horizontales
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

edges_x = convolve(img.astype(float), sobel_x)
edges_y = convolve(img.astype(float), sobel_y)
edges = np.sqrt(edges_x**2 + edges_y**2)  # magnitud del gradiente

plt.imshow(edges, cmap='gray')
plt.title("Bordes detectados con Sobel")
plt.show()
```

> **Pregunta**: ¿qué pasa si aplicás Sobel a la imagen sin suavizar? Vas a detectar bordes pero también mucho ruido (píxeles aislados que parecen bordes). Por eso el orden es: **suavizar primero, detectar bordes después**.

### 4.3 Canny Edge Detector

Canny es el detector de bordes clásico más robusto. Combina suavizado, gradiente, supresión de no-máximos, y doble umbralizado.

```python
from skimage.feature import canny

edges_canny = canny(img, sigma=1.0, low_threshold=0.1, high_threshold=0.3)

print(f"Píxeles de borde: {edges_canny.sum()}/{edges_canny.size} "
      f"({100*edges_canny.sum()/edges_canny.size:.1f}%)")
```

**Probar distintos thresholds**:

| low_threshold | high_threshold | Efecto |
|---------------|----------------|--------|
| 0.05 | 0.15 | Muchos bordes (ruidoso) |
| 0.1 | 0.3 | Balanceado (recomendado) |
| 0.2 | 0.5 | Pocos bordes (solo los fuertes) |

## 5. Transformaciones

### 5.1 Ecualización de histograma

Distribuye los valores de intensidad para usar todo el rango 0-255. Ideal para imágenes oscuras o con poco contraste.

```python
from skimage.exposure import equalize_hist

img_eq = equalize_hist(img)  # resultado en float [0, 1]

print(f"Antes: min={img.min()}, max={img.max()}")
print(f"Después: min={img_eq.min():.2f}, max={img_eq.max():.2f}")
```

**Salida esperada:**
```
Antes: min=23, max=211
Después: min=0.00, max=1.00
```

La imagen ahora usa todo el rango dinámico. El texto se vuelve más legible.

### 5.2 Transformada de Fourier

Descompone la imagen en ondas de diferentes frecuencias:

- **Bajas frecuencias**: regiones uniformes (fondos, cielos)
- **Altas frecuencias**: bordes, textura, ruido

```python
from scipy.fft import fft2, fftshift

F = fftshift(fft2(img))
magnitud = np.log(np.abs(F) + 1)  # escala logarítmica para visualizar

plt.figure(figsize=(12, 4))
plt.subplot(121); plt.imshow(magnitud, cmap='gray')
plt.title("Espectro de Fourier")
# El centro = frecuencias bajas. Lejos del centro = frecuencias altas
```

**Qué observar**: si la imagen tiene bordes nítidos, el espectro muestra un patrón de cruz brillante (líneas verticales/horizontales). Si está borrosa, el espectro se concentra en el centro.

## 6. Segmentación clásica

### 6.1 Umbralizado con Otsu

Otsu encuentra automáticamente el umbral óptimo que separa la imagen en dos clases (fondo y objeto), minimizando la varianza intra-clase.

```python
from skimage.filters import threshold_otsu

thresh = threshold_otsu(img)
binaria = img > thresh

print(f"Umbral encontrado: {thresh}")
print(f"Píxeles de foreground: {binaria.sum()}/{binaria.size}")
```

**Salida esperada:**
```
Umbral encontrado: 127
Píxeles de foreground: 320000/786432
```

### 6.2 Watershed

Trata la imagen como un mapa topográfico: los mínimos locales son "semillas" que se inundan. Útil para separar objetos que se tocan.

```python
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

# Distancia desde cada píxel de foreground al fondo más cercano
dist = distance_transform_edt(binaria)
# Los picos de distancia son los centros de cada objeto
markers = find_peaks(dist)  # simplificado
segmentos = watershed(-dist, markers)
```

## 7. Pipeline completo: extraer texto de un documento

```python
# 1. Cargar
img = np.array(Image.open('documento.jpg').convert('L'))

# 2. Corregir iluminación
img_eq = equalize_hist(img)

# 3. Suavizar ruido
img_suave = gaussian_filter(img_eq, sigma=1.0)

# 4. Binarizar con Otsu
thresh = threshold_otsu(img_suave)
binaria = img_suave > thresh

# 5. Detectar bordes (opcional, para guiar OCR)
bordes = canny(img_suave, sigma=0.5)

# Resultado: binaria tiene el texto listo para OCR
```

## 8. Common Mistakes

1. **No convertir a float antes de convolucionar**: `uint8` wrappea en 0-255. `convolve(img_uint8, kernel)` puede producir valores negativos que se truncan a 0. Siempre: `img.astype(float)`.

2. **Aplicar Canny sin suavizar**: el ruido de la cámara se detecta como bordes. Siempre Gaussian blur primero.

3. **Threshold fijo en vez de Otsu**: 127 no funciona para todas las imágenes. Otsu se adapta a la distribución de intensidades.

4. **Confundir magnitud y fase en Fourier**: la magnitud dice "cuánta energía" tiene cada frecuencia. La fase dice "dónde están" los bordes. Si reconstruís solo con magnitud, perdés la estructura de la imagen.

## 9. Check Your Understanding

1. Tenés una imagen con ruido sal-y-pimienta (píxeles blancos/negros aislados). ¿Qué filtro aplicás primero y por qué?
2. Canny detecta bordes que no existen en la realidad (falsos positivos). ¿Ajustás low_threshold hacia arriba o hacia abajo?
3. La ecualización de histograma mejora el contraste. ¿Hay casos donde empeore la imagen? ¿Cuáles?
4. Si la transformada de Fourier de una imagen tiene toda su energía en el centro, ¿qué me decís de la imagen?

## 10. Resumen

Las imágenes son matrices de números. Procesarlas es aplicar operaciones matemáticas sobre esas matrices: filtros de convolución para suavizar o detectar bordes, transformaciones de histograma para corregir iluminación, y umbralizado para segmentar. El pipeline clásico (suavizar → bordes → segmentar) es la base sobre la que se construyen sistemas modernos de visión. Dominarlo te permite debuguear pipelines de deep learning y resolver problemas sin GPU.

## 11. Where to Go Next

- [[Object Detection & Segmentation]] — pasar de bordes a bounding boxes
- [[CNNs]] — cómo la convolución clásica inspiró las redes convolucionales
- [[Generative Models for Vision]] — generación de imágenes con diffusion
- [[Image Processing Fundamentals]] — Fourier y filtrado avanzado
- [[Python Fundamentals]] — numpy para manipulación de arrays
- [[Feature Engineering]] — extracción de características desde imágenes
- [[Training Techniques]] — aumento de datos y preprocessing
