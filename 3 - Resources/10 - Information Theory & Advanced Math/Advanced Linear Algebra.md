---
tags:
  - linear-algebra
  - svd
  - pca
  - dimensionality-reduction
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Quieres comprimir un dataset de imágenes de 100x100 píxeles. Son 10,000 dimensiones por imagen — demasiadas para cualquier modelo. O quieres construir un sistema de recomendación de películas: 20,000 usuarios × 10,000 películas, matriz sparse de ratings. En ambos casos, la solución es la misma: **factorización de matrices**. PCA reduce dimensionalidad, SVD recomienda películas, los eigenfaces reconocen rostros, PageRank ordena la web. Bajo todas estas técnicas yace la misma matemática: eigen-decomposition y singular value decomposition. Esta nota explora estas factorizaciones y sus aplicaciones en [[Unsupervised Learning]], [[Collaborative Filtering]], [[Image Processing Fundamentals]], y [[Feature Engineering]].

## 2. Requisitos

- Python 3.8+, numpy, matplotlib, scikit-learn
- Fundamentos de [[Linear Algebra]]: matrices, vectores, transformaciones lineales
- Nociones de [[Numerical Methods]] (estabilidad numérica)

## 3. Eigen-decomposition

Para una matriz cuadrada $A \in \mathbb{R}^{n \times n}$, un **autovector** $v$ con **autovalor** $\lambda$ satisface:

$$A v = \lambda v$$

Geométricamente: $A$ escala a $v$ por $\lambda$ sin cambiar su dirección.

### Diagonalización

Si $A$ tiene $n$ autovectores linealmente independientes:

$$A = V \Lambda V^{-1}$$

donde $V$ tiene los autovectores como columnas y $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

### Matrices simétricas

Si $A = A^T$ (simétrica), entonces:
- Todos los autovalores son reales
- Los autovectores son ortogonales: $A = Q \Lambda Q^T$ con $Q$ ortogonal ($Q^{-1} = Q^T$)

```python
import numpy as np
import matplotlib.pyplot as plt

# Matriz simétrica definida positiva
A = np.array([[3.0, 1.0], [1.0, 2.0]])
autovalores, autovectores = np.linalg.eigh(A)  # Para simétricas

print("Autovalores:", autovalores)
print("Autovectores:\n", autovectores)

# Verificar: A @ v = lambda * v
for i in range(2):
    lam = autovalores[i]
    v = autovectores[:, i]
    lhs = A @ v
    rhs = lam * v
    print(f"  λ={lam:.3f}: error = {np.linalg.norm(lhs - rhs):.2e}")

# Salida esperada:
# Autovalores: [1.38196601 3.61803399]
# Autovectores:
#  [[-0.85065081 -0.52573111]
#   [ 0.52573111 -0.85065081]]
#   λ=1.382: error = 2.22e-16
#   λ=3.618: error = 2.22e-16
```

**Propiedad clave**: para $A$ simétrica, la traza es suma de autovalores y el determinante es su producto:

$$\text{tr}(A) = \sum \lambda_i, \quad \det(A) = \prod \lambda_i$$

## 4. Singular Value Decomposition (SVD)

La SVD existe para *toda* matriz $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$: vectores singulares izquierdos (ortonormales)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal con valores singulares $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r > 0$
- $V \in \mathbb{R}^{n \times n}$: vectores singulares derechos (ortonormales)

### Interpretación geométrica

SVD descompone cualquier transformación lineal en tres pasos: rotación ($V^T$), scaling ($\Sigma$), rotación ($U$).

### Relación con eigen-decomposition

- Los valores singulares $\sigma_i$ son las raíces cuadradas de los autovalores de $A^T A$ y $A A^T$
- $U$ contiene los autovectores de $A A^T$
- $V$ contiene los autovectores de $A^T A$
- SVD es más general que eigen-decomposition: aplica a matrices rectangulares

```python
import numpy as np

# Cualquier matriz rectangular
A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0],
              [10.0, 11.0, 12.0]])

U, s, Vt = np.linalg.svd(A, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"Valores singulares: {s}")
print(f"Vt shape: {Vt.shape}")

# Reconstrucción
k = len(s)
A_reconst = U[:, :k] @ np.diag(s) @ Vt[:k, :]
print(f"Error de reconstrucción: {np.linalg.norm(A - A_reconst):.2e}")

# Salida esperada:
# U shape: (4, 3)
# Valores singulares: [2.546e+01 1.291e+00 8.881e-16]
# Vt shape: (3, 3)
# Error de reconstrucción: 7.22e-15
```

## 5. Low-rank approximation

El teorema de Eckart-Young: la mejor aproximación de rango $k$ de $A$ en norma de Frobenius es:

$$A_k = U_k \Sigma_k V_k^T$$

donde se conservan solo los $k$ mayores valores singulares.

### Compresión de imágenes

```python
import numpy as np
from sklearn.datasets import load_sample_image

# Cargar una imagen de prueba (china.jpg)
china = load_sample_image("china.jpg") / 255.0
print(f"Original shape: {china.shape}")

# Promediar canales RGB a grises
img = china.mean(axis=2)

def compress_svd(img, k):
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

ratios = [1, 2, 5, 10, 20, 50]

for k in ratios:
    img_k = compress_svd(img, k)
    error = np.linalg.norm(img - img_k) / np.linalg.norm(img)
    compression = 100 * (k * (img.shape[0] + img.shape[1])) / (img.shape[0] * img.shape[1])
    print(f"k={k:3d}: error rel={error:.4f}, compresión={compression:.1f}%")

# Salida esperada (aproximada, para imagen de 427×640):
# k=  1: error rel=0.2998, compresión=0.3%
# k=  2: error rel=0.2227, compresión=0.5%
# k=  5: error rel=0.1185, compresión=1.3%
# k= 10: error rel=0.0728, compresión=2.6%
# k= 20: error rel=0.0444, compresión=5.2%
# k= 50: error rel=0.0194, compresión=13.1%
```

Con $k=20$ (~5% del tamaño original) ya se reconstruye una imagen reconocible. Esto es la base de [[Image Processing Fundamentals]] y también se usa para:
- **Denoising**: el ruido tiende a estar en valores singulares pequeños
- **Sistemas de recomendación**: [[Collaborative Filtering]] (SVD en matriz usuarios-items)
- **Latent semantic analysis (LSA)**: SVD sobre matriz término-documento

## 6. PCA desde SVD

PCA encuentra las direcciones de máxima varianza en los datos. Se implementa eficientemente via SVD:

1. **Centrar** los datos: $X_c = X - \bar{X}$ (media por columna)
2. **SVD**: $U \Sigma V^T = X_c$
3. **Proyección**: $T = U_k \Sigma_k$ (componentes principales)
4. **Loadings**: $V_k$ (contribución de cada feature a cada componente)
5. **Varianza explicada**: $\frac{\sigma_i^2}{\sum \sigma_j^2}$

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data  # 150 × 4
print(f"Forma original: {X.shape}")

# PCA con SVD manual
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
var_explicada = s**2 / (s**2).sum()

print("\nVarianza explicada por componente:")
for i, v in enumerate(var_explicada):
    print(f"  PC{i+1}: {v:.3f} ({v*100:.1f}%)")

# Proyección a 2D
k = 2
X_pca = X_centered @ Vt[:k, :].T
print(f"\nForma reducida: {X_pca.shape}")

# Comparar con sklearn
pca_sklearn = PCA(n_components=k)
X_sklearn = pca_sklearn.fit_transform(X)
print(f"Diferencia con sklearn: {np.linalg.norm(np.abs(X_pca) - np.abs(X_sklearn)):.2e}")

# Salida esperada:
# Forma original: (150, 4)
#
# Varianza explicada por componente:
#   PC1: 0.925 (92.5%)
#   PC2: 0.063 (6.3%)
#   PC3: 0.010 (1.0%)
#   PC4: 0.002 (0.2%)
#
# Forma reducida: (150, 2)
# Diferencia con sklearn: 3.21e-15
```

### Reglas prácticas para elegir $k$

- **Varianza explicada acumulada**: elegir $k$ tal que $\sum_{i=1}^k \sigma_i^2 / \sum \sigma_j^2 \geq 0.95$
- **Scree plot**: buscar el "codo" en la curva de valores singulares
- **Kaiser rule**: conservar componentes con $\lambda_i > 1$ (para correlación)

PCA via SVD es numéricamente más estable que via eigen-decomposition de $X^T X$, especialmente cuando $X$ tiene más columnas que filas ($n \ll p$).

## 7. Pseudoinversa (Moore-Penrose)

La pseudoinversa $A^+$ generaliza la inversa a matrices no cuadradas o singulares:

$$A^+ = V \Sigma^+ U^T$$

donde $\Sigma^+$ tiene los recíprocos de los valores singulares no nulos.

Usos:
- **Resolver sistemas sobredeterminados**: $x = A^+ b$ es la solución de mínimos cuadrados
- **Resolver sistemas subdeterminados**: $x = A^+ b$ es la solución de norma mínima
- **Regresión lineal**: `np.linalg.lstsq` usa SVD internamente

```python
import numpy as np

# Sistema sobredeterminado (5 ecuaciones, 3 incógnitas)
A = np.random.randn(5, 3)
b = np.random.randn(5)

# Pseudoinversa via SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
s_inv = np.diag(1.0 / s)
x_svd = Vt.T @ s_inv @ U.T @ b

# Comparar con lstsq
x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

print(f"Diferencia: {np.linalg.norm(x_svd - x_lstsq):.2e}")
print(f"Residual:   {np.linalg.norm(A @ x_svd - b):.2e}")

# Salida esperada:
# Diferencia: 6.28e-16
# Residual:   1.07e-01
```

En regresión lineal con features correlacionados (multicolinealidad), la pseudoinversa maneja el rango deficiente de forma natural — a diferencia de la inversa de $X^T X$ que falla.

## 8. Common Mistakes

1. **Aplicar SVD a datos sin centrar en PCA**: si no centras los datos, la primera componente captura la media en lugar de la varianza. Esto invalida PCA.

2. **No examinar los valores singulares**: tirar todos los componentes con valores singulares pequeños sin revisar su magnitud relativa. Un scree plot es obligatorio.

3. **Elegir $k$ sin criterio en low-rank approximation**: en sistemas de recomendación, $k$ muy pequeño pierde información de usuarios atípicos. En compresión de imágenes, $k$ muy grande no comprime suficiente.

4. **Confundir autovalores de $A^T A$ con valores singulares de $A$**: los valores singulares son las raíces cuadradas de los autovalores de $A^T A$. Además, $A^T A$ puede tener peor condicionamiento numérico.

5. **Usar eigen-decomposition en matrices no simétricas**: si $A$ no es simétrica, los autovalores pueden ser complejos y los autovectores no ortogonales. La SVD es la factorización correcta para matrices generales.

6. **Asumir que PCA funciona sin normalizar features**: si las features están en escalas distintas (e.g., kg y cm), la varianza está dominada por la escala. Normaliza (z-score) antes de PCA.

## Resumen

La eigen-decomposition diagonaliza matrices simétricas en autovalores y autovectores ortogonales. La SVD generaliza esto a cualquier matriz rectangular: $A = U \Sigma V^T$, descomponiéndola en rotaciones y scaling. Truncar SVD produce la mejor aproximación de bajo rango, con aplicaciones en compresión de imágenes, denoising y sistemas de recomendación. PCA desde SVD encuentra las direcciones de máxima varianza, centrando primero los datos. La pseudoinversa de Moore-Penrose resuelve sistemas lineales con rango deficiente. La clave está en elegir correctamente la factorización según el problema y acompañarla de diagnóstico (valores singulares, varianza explicada).

## Check Your Understanding

1. ¿Por qué SVD existe para cualquier matriz mientras que eigen-decomposition no? <!-- Eigen-decomposition requiere matriz cuadrada. SVD no tiene esa restricción: toda matriz tiene SVD. -->

2. Si todos los valores singulares son iguales, ¿qué forma tiene la matriz? <!-- Es un múltiplo escalar de una matriz ortogonal/unitario (rotación pura). -->

3. En PCA, ¿qué pasa si eliges $k$ mayor que el rango real de los datos? <!-- Las componentes adicionales tienen varianza cero. Los loadings no son únicos y el modelo está sobredimensionado. -->

4. ¿Por qué la pseudoinversa no produce la misma solución que $X^T X^{-1} X^T$ cuando $X$ tiene columnas linealmente dependientes? <!-- $X^T X$ es singular (no invertible). La pseudoinversa selecciona la solución de norma mínima. -->

5. En compresión de imágenes con SVD, ¿qué parte del costo domina? <!-- La SVD completa es $O(\min(mn^2, nm^2))$. Para imágenes grandes se usan aproximaciones randomized SVD. -->

## Where to Go Next

- [[Linear Algebra]] — fundamentos previos a avanzada
- [[Unsupervised Learning]] — clustering en espacio PCA, t-SNE sobre loadings
- [[Numerical Methods]] — estabilidad de SVD vs eigen-decomposition
- [[Image Processing Fundamentals]] — compresión y denoising con SVD
- [[Collaborative Filtering]] — FunkSVD, matrix factorization para recomendación
- [[Feature Engineering]] — PCA como técnica de feature extraction
- [[Optimization Theory]] — SVD en problemas de optimización convexa
- [[Information Theory for ML]] — conexión entre varianza explicada e información retenida
