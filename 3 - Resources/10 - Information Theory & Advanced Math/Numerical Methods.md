---
tags:
  - numerical-methods
  - floating-point
  - stability
  - optimization
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Ejecutas una regresión lineal con `np.linalg.inv(X.T @ X) @ X.T @ y` y obtienes `LinAlgError: Singular matrix`. O entrenas una red neuronal y la loss se vuelve `NaN` en la época 100 sin razón aparente. No son bugs de lógica — son **problemas numéricos**. El hardware no puede representar números reales con precisión infinita. [[Linear Algebra]] te da las operaciones; [[Numerical Methods]] te enseña cómo ejecutarlas de forma estable. Esta nota cubre representación floating point, catastrophic cancellation, log-sum-exp trick, y cómo elegir la factorización matricial correcta para cada problema.

## 2. Requisitos

- Python 3.8+, numpy, scipy.linalg
- Familiaridad con [[Linear Algebra]]: matrices, sistemas lineales, factorizaciones
- Conceptos de [[Calculus]] y [[Gradient-Based Optimization]]

## 3. Floating Point: cómo miente tu computadora

IEEE 754 define la representación de punto flotante:

$$x = (-1)^s \times m \times 2^{e - \text{bias}}$$

- **Precisión simple (float32)**: 23 bits mantisa → ~7 decimales
- **Precisión doble (float64)**: 52 bits mantisa → ~16 decimales
- **Machine epsilon ($\epsilon$)**: distancia entre 1.0 y el siguiente número representable

```python
import numpy as np

print(f"float32 epsilon: {np.finfo(np.float32).eps:.2e}")
print(f"float64 epsilon: {np.finfo(np.float64).eps:.2e}")
print(f"Máximo float64:  {np.finfo(np.float64).max:.2e}")
print(f"Mínimo positivo:  {np.finfo(np.float64).tiny:.2e}")

# Overflow
print(f"Overflow: {np.float64(1e308) * 10:.2e}")

# Underflow
print(f"Underflow: {np.float64(1e-308) / 10:.2e}")

# Salida esperada:
# float32 epsilon: 1.19e-07
# float64 epsilon: 2.22e-16
# Máximo float64:  1.80e+308
# Mínimo positivo:  2.23e-308
# Overflow: inf
# Underflow: 0.00e+00
```

Problemas fundamentales:

- **Overflow**: números que exceden el máximo → `inf`
- **Underflow**: números por debajo del mínimo → `0` (pérdida total de información)
- **Rounding**: la mayoría de números reales no son representables exactamente
- **Catastrophic cancellation**: pérdida de dígitos significativos al restar números casi iguales

## 4. Catastrophic Cancellation

Ocurre cuando restas dos números aproximadamente iguales. Los dígitos significativos se cancelan y solo queda ruido de redondeo.

```python
import numpy as np

# Ejemplo clásico: fórmula cuadrática
# x^2 - 1000000.000001x + 1 = 0
# Una raíz es ~ 1e-6, la otra ~ 1e6

a, b, c = 1.0, -1000000.000001, 1.0

# Fórmula inestable
x1_inestable = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
print(f"Raíz pequeña (inestable): {x1_inestable:.12f}")

# Fórmula estable: usar x1 = c / (a * x2)
x2 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
x1_estable = c / (a * x2)
print(f"Raíz pequeña (estable):   {x1_estable:.12f}")

# Salida esperada:
# Raíz pequeña (inestable): 0.000000000000
# Raíz pequeña (estable):   0.000001000000
```

**Ejemplo crítico en ML: varianza**

La fórmula ingenua $\sigma^2 = \frac{1}{n}\sum x_i^2 - \bar{x}^2$ sufre cancellation cuando los datos tienen media grande y varianza pequeña (típico en imágenes con píxeles $[0,255]$).

```python
import numpy as np

# Datos con media grande y varianza pequeña
x = np.array([1000.0, 1000.1, 999.9, 1000.05, 999.95])

# Varianza inestable
var_inestable = np.mean(x**2) - np.mean(x)**2
print(f"Varianza (inestable):   {var_inestable:.10f}")

# Varianza estable (Welford)
var_estable = np.var(x, ddof=0)
print(f"Varianza (estable):     {var_estable:.10f}")

# Salida esperada:
# Varianza (inestable):   0.0030000000
# Varianza (estable):     0.0030000000
```

En la práctica, siempre usa `np.var` o `np.std` que implementan el algoritmo de dos pasadas estable.

**Softmax naive**: $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

Si $z_i$ tiene valores grandes (e.g., $z = [1000, 1001, 999]$), $e^{1000}$ produce overflow inmediato.

## 5. Log-Sum-Exp Trick

Estabiliza el cálculo de $\log(\sum e^{z_i})$ extrayendo el máximo:

$$\log\sum_{j} e^{z_j} = \max(z) + \log\sum_{j} e^{z_j - \max(z)}$$

Aplicado a softmax y cross-entropy:

$$\log p_i = z_i - \max(z) - \log\sum_j e^{z_j - \max(z)}$$

```python
import numpy as np

def softmax_naive(z):
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()

def softmax_stable(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

def cross_entropy_stable(y_true, logits):
    # logits son las salidas pre-softmax
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs -= np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
    return -np.sum(y_true * log_probs) / len(y_true)

# Logits con valores extremos
z = np.array([1000.0, 1010.0, 990.0])

try:
    prob_naive = softmax_naive(z)
    print(f"Softmax naive: {prob_naive}")
except FloatingPointError as e:
    print(f"Softmax naive: overflow!")

# Usar log-sum-exp
prob_stable = softmax_stable(z)
print(f"Softmax stable:              {prob_stable}")

# Cross-entropy directamente sobre logits
y_true = np.array([0.0, 1.0, 0.0])
ce = cross_entropy_stable(y_true.reshape(1, -1), z.reshape(1, -1))
print(f"Cross-entropy (estable):     {ce:.6f}")

# Salida esperada:
# Softmax naive: [nan nan nan]
# Softmax stable:              [4.53978686e-05 9.99909216e-01 6.13972499e-09]
# Cross-entropy (estable):     0.0000908
```

Este truco es estándar: `torch.nn.CrossEntropyLoss` acepta logits directamente porque aplica log-sum-exp internamente.

## 6. Resolución de sistemas lineales

Dado $Ax = b$, hay múltiples formas de obtener $x$. No todas son igual de estables.

| Método | Costo | Cuándo usarlo |
|--------|-------|---------------|
| `x = np.linalg.inv(A) @ b` | $O(n^3)$ | **Nunca**. Inestable e ineficiente. |
| `x = np.linalg.solve(A, b)` | $O(n^3)$ | Sistemas densos cuadrados. Usa LU con pivoting. |
| `x = scipy.linalg.solve(A, b, assume_a='pos')` | $O(n^3/3)$ | Si $A$ es simétrica definida positiva (Cholesky). |
| `x = np.linalg.lstsq(A, b)` | $O(n^3)$ | Sistemas sobredeterminados o subdeterminados. Usa SVD. |
| `x = scipy.sparse.linalg.spsolve(A, b)` | — | Matrices sparse grandes. |

### Condition Number

El número de condición $\kappa(A) = \|A\|\|A^{-1}\|$ mide cuán sensible es la solución a errores en los datos:

- $\kappa(A)$ cercano a 1: bien condicionado
- $\kappa(A)$ grande ($>10^6$): mal condicionado, pequeñas perturbaciones en $b$ producen grandes cambios en $x$

```python
import numpy as np
import scipy.linalg

# Matriz de Hilbert (mal condicionada)
A = scipy.linalg.hilbert(10)
b = np.ones(10)

kappa = np.linalg.cond(A)
print(f"Número de condición de A: {kappa:.2e}")

# Resolver con distintos métodos
x_solve = np.linalg.solve(A, b)
x_inv   = np.linalg.inv(A) @ b

# Verificar error residual
print(f"Error solve: {np.linalg.norm(A @ x_solve - b):.2e}")
print(f"Error inv:   {np.linalg.norm(A @ x_inv - b):.2e}")

# Salida esperada:
# Número de condición de A: 1.60e+13
# Error solve: 2.35e-10
# Error inv:   8.72e-10
```

La matriz de Hilbert con $n=10$ tiene $\kappa \approx 10^{13}$. Cualquier método sufre, pero `solve` con LU pivoteado es más estable que usar la inversa.

## 7. Implementaciones estables

### log1p y expm1

Para valores pequeños: $\log(1+x)$ y $e^x - 1$ pierden precisión cerca de 0.

```python
import numpy as np

x = 1e-15

# Inestable
naive_log = np.log(1 + x)
naive_exp = np.exp(x) - 1

# Estable
stable_log = np.log1p(x)
stable_exp = np.expm1(x)

print(f"log(1 + x):        {naive_log:.20f}")
print(f"log1p(x):          {stable_log:.20f}")
print(f"exp(x) - 1:        {naive_exp:.20f}")
print(f"expm1(x):          {stable_exp:.20f}")

# Salida esperada:
# log(1 + x):        0.00000000000000000000
# log1p(x):          0.00000000000000100000
# exp(x) - 1:        0.00000000000000000000
# expm1(x):          0.00000000000000100000
```

### Evitar la inversa explícita

En regresión lineal, $\beta = (X^T X)^{-1} X^T y$ nunca debe calcularse con `inv`:

```python
import numpy as np

# MAL: inestable e ineficiente
beta_mal = np.linalg.inv(X.T @ X) @ X.T @ y

# BIEN: solve para el sistema normal (solo para X^T X bien condicionada)
beta_bien = np.linalg.solve(X.T @ X, X.T @ y)

# MEJOR: lstsq usa SVD, maneja rank deficiency
beta_mejor, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

# MEJOR AÚN: scipy.linalg.lstsq con SVD truncado si es necesario
```

## 8. Common Mistakes

1. **Comparar floats con `==`**: nunca uses `x == 0.0` después de operaciones. Usa `np.isclose(x, 0.0)` o `abs(x) < tol`.

2. **No usar log-sum-exp en cross-entropy**: pasar probabilidades (post-softmax) a la loss function en lugar de logits. Esto multiplica los errores de redondeo.

3. **Ignorar el condition number**: resolver $Ax=b$ con $\kappa(A) > 10^{10}$ produce soluciones sin significado físico. Siempre verifica `np.linalg.cond(A)`.

4. **Usar `np.linalg.inv` para resolver sistemas**: es $2$ a $3$ veces más costoso y numéricamente menos estable que `np.linalg.solve`.

5. **No distinguir float32 de float64**: en PyTorch/TensorFlow, el default es float32. Las acumulaciones de gradiente pueden underflow. Usa float32 para entrenamiento (rápido), float64 para validación numérica.

6. **Calcular varianza con la fórmula ingenua**: `np.mean(x**2) - np.mean(x)**2` puede dar negativa por errores de redondeo. Siempre usa `np.var`.

## Resumen

La estabilidad numérica es tan importante como la corrección matemática. El estándar IEEE 754 impone límites finitos que producen overflow, underflow y rounding errors. La catastrophic cancellation ocurre al restar números casi iguales y corrompe la varianza naive y otras fórmulas. El log-sum-exp trick estabiliza softmax y cross-entropy extrayendo el máximo. Para sistemas lineales, la elección de factorización (LU, Cholesky, QR, SVD) y verificar el condition number es crucial. Funciones como `log1p`, `expm1`, y evitar la inversa explícita con `np.linalg.solve` o `np.linalg.lstsq` son prácticas esenciales. En ML, la mayoría de los NaN no son bugs lógicos sino numéricos — y se resuelven con estas técnicas.

## Check Your Understanding

1. ¿Por qué `np.log(1 + 1e-16)` da 0.0 mientras que `np.log1p(1e-16)` no? <!-- 1 + 1e-16 se redondea a 1.0 en float64 porque el epsilon es ~2e-16. log1p evita la suma intermedia. -->

2. Si un sistema lineal tiene $\kappa(A) = 10^{12}$, ¿cuántos dígitos de precisión puedes esperar en $x$? <!-- En float64 (~16 dígitos), pierdes ~12, quedan ~4 dígitos confiables. -->

3. ¿Por qué la cross-entropy debería recibir logits y no probabilidades? <!-- Si recibe probabilidades post-softmax, el log-sum-exp ya se aplicó en el softmax, pero si el usuario pasa las probabilidades por separado, se pierde la estabilidad del log-sum-exp combinado. -->

4. ¿Qué ventaja tiene Cholesky sobre LU para matrices simétricas definidas positivas? <!-- La mitad del costo ($n^3/3$ vs $2n^3/3$), mayor estabilidad, y garantiza que la matriz sea SPD. -->

5. ¿Puede la varianza calculada con la fórmula ingenua dar un valor negativo? <!-- Sí. Si $\sum x_i^2$ y $n\bar{x}^2$ son casi iguales, la cancelación catastrófica puede producir un resultado negativo. -->

## Where to Go Next

- [[Linear Algebra]] — fundamentos de matrices y sistemas lineales
- [[Calculus]] — series de Taylor para entender errores de aproximación
- [[Gradient-Based Optimization]] — cómo la estabilidad numérica afecta el entrenamiento
- [[Training Techniques]] — mixed precision training, gradient scaling
- [[Probability]] — distribución de errores numéricos
- [[Python for Data Science]] — herramientas numéricas en el ecosistema Python
- [[Convex Optimization]] — problemas mal condicionados en optimización
- [[Advanced Linear Algebra]] — SVD como herramienta para sistemas mal condicionados
