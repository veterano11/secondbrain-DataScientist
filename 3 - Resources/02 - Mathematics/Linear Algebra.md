---
tags: [mathematics, linear-algebra, foundational]
status: growing
created: 2026-06-27
---

# Álgebra Lineal

## 1. Escenario de aprendizaje

Estás entrenando un modelo de transformers para generar texto. Cada token que produces es un vector, y el mecanismo de atención no es más que una serie de multiplicaciones de matrices. Cuando ejecutas PCA para reducir la dimensionalidad de tus datos, estás descomponiendo una matriz de covarianza en sus componentes fundamentales. Sin álgebra lineal, los modelos son cajas negras. Con ella, ves que todo son vectores siendo transformados.

Cada vez que entrenas un modelo de ML, estás haciendo álgebra lineal. El álgebra lineal **es el lenguaje en el que está escrito el machine learning**. En esta nota construimos desde vectores simples hasta las descomposiciones más poderosas, siempre conectando con cómo se usan en ML.

---

## 2. Vectores

### 2.1 Intuición

Un vector es una **flecha en el espacio** con una dirección y una magnitud. Pero también es una **lista ordenada de números**. Ambas visiones son equivalentes.

Imagina un punto en un plano cartesiano 2D. El vector $v = (3, 4)$ significa "camina 3 unidades a la derecha y 4 hacia arriba". Eso es un vector.

Cada fila de tu conjunto de datos $X$ es un vector. Si tu dataset tiene 100 personas con 5 mediciones (edad, altura, ingresos, educación, horas_dormidas), cada persona es un vector en $\mathbb{R}^5$. No puedes visualizar 5 dimensiones, pero matemáticamente funciona exactamente igual que 2 o 3.

### 2.2 Definición formal y operaciones

Un vector $v \in \mathbb{R}^n$ es una tupla ordenada de $n$ números reales.

**Suma**: $v + w = (v_1 + w_1, v_2 + w_2, ...)$ — componente por componente. Coloca una flecha en la punta de la otra.

**Multiplicación escalar**: $c \cdot v = (c \cdot v_1, c \cdot v_2, ...)$ — estira o encoge la flecha. Si $c < 0$, invierte la dirección.

### 2.3 El producto punto

Aquí es donde comienza la magia.

$$v \cdot w = \sum_{i=1}^n v_i w_i = v_1 w_1 + v_2 w_2 + ... + v_n w_n$$

Equivalentemente:

$$v \cdot w = \|v\| \cdot \|w\| \cdot \cos\theta$$

donde $\theta$ es el ángulo entre los vectores.

**Ejemplo concreto**:
```
v = [3, 4]
w = [1, 2]

v · w = 3×1 + 4×2 = 3 + 8 = 11
‖v‖ = √(3² + 4²) = √25 = 5
‖w‖ = √(1² + 2²) = √5 ≈ 2.236

cos θ = 11 / (5 × 2.236) = 11 / 11.18 ≈ 0.984
θ ≈ 10.3°
```

**¿Qué nos dice el producto punto?**

- **Mide similitud**: si dos vectores apuntan en la misma dirección, el producto punto es grande y positivo. Opuesto → negativo. Perpendicular → **cero**.
- **Es el corazón de la atención en transformers**: $QK^T$ es una matriz de productos punto entre queries y keys. Cada celda $(i, j)$ dice "¿cuánto se relaciona el token $i$ con el token $j$?"
- **Es la base de la similitud por coseno**: $\text{cosine sim}(v, w) = \frac{v \cdot w}{\|v\| \|w\|}$, que es lo que usas en RAG para encontrar documentos similares.

### 2.4 Norma (Magnitud)

La norma L2 (la más común) es la longitud de la flecha:

$$\|v\|_2 = \sqrt{\sum v_i^2}$$

Para $v = [3, 4]$: $\|v\| = \sqrt{9 + 16} = 5$. Esta es la hipotenusa de un triángulo 3-4-5.

**Norma L1**: $\|v\|_1 = \sum |v_i| = 3 + 4 = 7$. Se usa en regularización Lasso — lleva los pesos exactamente a cero.

**¿Por qué normalizar?** Si una característica es "edad" (0-100) y otra es "ingresos" (0-1,000,000), el producto punto estará dominado por los ingresos. Normalizar (vectores unitarios donde $\|v\| = 1$) asegura que todas las características pesen por igual.

### 2.5 Independencia lineal

Un conjunto de vectores es **linealmente independiente** si ningún vector puede escribirse como combinación de los demás.

```
v₁ = [1, 0]
v₂ = [0, 1]    → Independientes (forman una base)
v₃ = [2, 3]    → Dependiente de v₁ y v₂

Porque v₃ = 2·v₁ + 3·v₂
```

**¿Por qué importa esto?** En ML, la independencia lineal se relaciona con la **multicolinealidad**. Si dos características son linealmente dependientes (o casi), tu modelo tendrá coeficientes inestables. Por eso hacemos selección de características y regularización.

---

## 3. Matrices

### 3.1 Intuición

Una matriz es una **colección de vectores** organizados en filas y columnas. Pero también es una **transformación**: multiplicar una matriz por un vector rota, escala y refleja ese vector.

Tu conjunto de datos $X$ es una matriz $n \times d$: $n$ filas (muestras), $d$ columnas (características).

### 3.2 Multiplicación matriz-vector

$$A \cdot v = w$$

Cada elemento de $w$ es el producto punto de una fila de $A$ con $v$:

```
A = [1 2]    v = [3]    A·v = [1×3 + 2×4] = [11]
    [3 4]        [4]          [3×3 + 4×4]   [25]
```

**Visualización**: la matriz transforma el espacio. Si tomas cada punto de un cuadrado y lo multiplicas por una matriz, obtienes un paralelogramo rotado y estirado.

**En una red neuronal**: cada capa hace $h = Wx + b$. $W$ es una matriz que transforma el vector de entrada $x$ en el vector oculto $h$. Aprender es encontrar la $W$ correcta.

### 3.3 Multiplicación matriz-matriz

$$C = A \cdot B$$

Cada columna de $C$ es $A$ por la columna correspondiente de $B$. O: $C_{ij} = \text{fila}_i(A) \cdot \text{col}_j(B)$.

**¿Por qué definirla así?** Porque representa **composición de transformaciones**. Primero transforma con $B$, luego con $A$. El orden importa: $AB \neq BA$ (no conmutativa).

### 3.4 Transpuesta e Inversa

- **Transpuesta** $A^T$: intercambia filas y columnas. $(A^T)_{ij} = A_{ji}$.
- **Inversa** $A^{-1}$: $A \cdot A^{-1} = I$ (identidad). Existe solo si $A$ es cuadrada y de **rango completo**.

**¿Qué hace la inversa?** Deshace la transformación. Si $w = A v$, entonces $v = A^{-1} w$. En regresión lineal: $\hat{w} = (X^T X)^{-1} X^T y$ — estamos "deshaciendo" la mezcla para encontrar los coeficientes.

### 3.5 Rango

El **rango** de una matriz es el número de filas/columnas linealmente independientes. Mide cuánta información contiene realmente la matriz.

- **Rango deficiente**: si tienes 1000 características pero muchas son combinaciones lineales de otras, el rango efectivo es mucho menor.
- **Rango completo**: cada fila/columna aporta información nueva.
- Una matriz de rango $r$ puede aproximarse por matrices de rango inferior (SVD da la mejor aproximación).

### 3.6 Traza

$$\text{tr}(A) = \sum_i A_{ii}$$

Suma de la diagonal. Aparece en propiedades como $\text{tr}(ABC) = \text{tr}(BCA)$ (cíclica) y en el cálculo de la varianza total explicada en PCA.

---

## 4. Descomposiciones — El corazón del álgebra lineal para ML

### 4.1 Eigenvalores y Eigenvectores

$$A v = \lambda v$$

$v$ es un **eigenvector** de $A$, y $\lambda$ es su **eigenvalor**.

**Intuición**: cuando aplicas la transformación $A$ a $v$, el resultado apunta en la **misma dirección**. Solo se estira (o encoge) por el factor $\lambda$.

Imagina una matriz que estira el espacio 2D: el eje X se duplica, el eje Y se mantiene. Los eigenvectors son $[1,0]$ (con $\lambda=2$) y $[0,1]$ (con $\lambda=1$).

**Aplicación directa — PCA**:
1. Calcula la matriz de covarianza $\Sigma = \frac{1}{n} X^T X$
2. Sus eigenvectors son las **direcciones de máxima varianza**
3. Sus eigenvalores indican **cuánta varianza explica cada dirección**
4. Toma los $k$ eigenvectors con mayor $\lambda$ y proyecta los datos → reducción de dimensionalidad

**Teorema Espectral**: si $A$ es simétrica ($A = A^T$, como la covarianza), sus eigenvectors son ortogonales. Esto hace todo más manejable.

### 4.2 Descomposición en Valores Singulares (SVD)

$$A = U \Sigma V^T$$

Esta es la **más poderosa** de todas las descomposiciones.

**Intuición geométrica**: toda transformación lineal se descompone en tres pasos:
1. **Rotación** ($V^T$) — reorienta el espacio
2. **Escalamiento** ($\Sigma$) — estira/encoge cada eje por su valor singular
3. **Otra rotación** ($U$) — rota al sistema de coordenadas final

**Ejemplo concreto**:
```
A = [3 1]
    [1 3]

SVD:
U = [0.707 -0.707]    Σ = [4 0]    V^T = [0.707  0.707]
    [0.707  0.707]        [0 2]          [-0.707 0.707]
```

Los valores singulares son $\sigma_1 = 4$, $\sigma_2 = 2$. El primero explica $\frac{4}{4+2} = 66.7\%$ de la "energía" de la matriz.

**¿Por qué es tan importante SVD?**

| Aplicación | Cómo usa SVD |
|---|---|
| **PCA** | PCA = SVD de la matriz centrada. Componentes = $V$, varianza = $\Sigma^2$ |
| **Recomendación** | Premio Netflix: factorizar matriz usuario-item con SVD truncado |
| **Compresión** | Conserva solo los $k$ valores singulares más grandes (10× más pequeño, calidad ligeramente inferior) |
| **Denoising** | El ruido vive en los componentes pequeños; eliminarlos = filtrar |
| **Embeddings de LLM** | Word2vec, GloVe — factorizar matrices de co-ocurrencia es SVD |
| **Completación de matrices** | Predecir entradas faltantes (como en recomendación) |

### 4.3 Descomposición de Cholesky

$$A = L L^T$$

Para matrices simétricas definidas positivas (como la covarianza).

**¿Por qué es útil?**
- Resuelve sistemas lineales rápido ($Ax = b$ en $O(n^2)$ con Cholesky vs $O(n^3)$ con inversa)
- Muestrear de distribuciones Gaussianas multivariadas
- Procesos Gaussianos

---

## 5. Common Mistakes

1. **Not standardizing before PCA**: PCA maximizes variance. If "age" (0-100) and "income" (0-1M) are on different scales, the first component will essentially be "income". Always standardize (mean 0, variance 1).

2. **Confusing eigenvectors of $X$ with eigenvectors of $X^T X$**: PCA operates on the covariance $X^T X$, not on $X$ directly. The eigenvectors of $X^T X$ are the principal directions.

3. **SVD is not PCA**: PCA is SVD applied to the centered matrix. If you do not center the data, SVD gives directions of maximum "energy" but not necessarily maximum variance.

4. **Non-invertible matrix ≠ useless matrix**: It means there is linear dependence. Regularization (Ridge) solves this.

5. **Forgetting that $QK^T$ in attention is a dot product matrix**: Each entry $(i,j)$ is $q_i \cdot k_j$. When $q_i$ and $k_j$ are orthogonal, attention is zero for that pair.

---

## 6. Check Your Understanding

1. Given two vectors $v = [1, 0]$ and $w = [0, 1]$, what is their dot product? What does this tell you about the angle between them?
2. Why is the covariance matrix $X^T X$ symmetric? And why are its eigenvectors orthogonal?
3. If you SVD a matrix and all singular values are equal, what shape is the matrix?
4. In a 2-layer neural network: $h = W_1 x$, $y = W_2 h$, what does each matrix $W$ represent geometrically?
5. You have 100 samples and 1000 features. The effective rank is 80. What does this mean and how does it affect your model?

---

## 7. Summary

Linear algebra is the language of machine learning. Your data is a collection of vectors. Transformations (matrices) are your models. Decompositions (SVD, eigenvalues) are the tools that let you see what is really happening. When you see a transformer, remember: $QK^T$ is a matrix of dot products measuring token similarity. When you run PCA, remember: you are finding the eigenvectors of the covariance. Everything is linear algebra.

---

## 8. Where to Go Next

- [[Calculus]] — Optimization requires derivatives
- [[Probability]] — Distributions live in vector spaces
- [[Statistics]] — Covariance is a linear algebra concept
- [[Neural Networks]] — Every layer is $Wx + b$
- [[Transformers]] — Self-attention = dot products between queries and keys
- [[Unsupervised Learning]] — PCA, t-SNE, K-Means all use linear algebra heavily
- [[Gradient-Based Optimization]] — Gradient descent updates are matrix operations
