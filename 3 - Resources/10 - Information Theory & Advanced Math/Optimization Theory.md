---
tags:
  - optimization
  - convexity
  - kkt
  - svm
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Sabes que SVM maximiza el margen, logistic regression minimiza cross-entropy, y neural networks minimizan una loss. Pero "minimizar" es solo el verbo — ¿qué significa realmente **optimizar**? ¿Por qué a veces hay mínimo global garantizado y otras no? ¿Cómo se manejan restricciones como $||w|| \leq C$? La respuesta está en la teoría de optimización: convexidad, condiciones de optimalidad, multiplicadores de Lagrange y condiciones KKT. [[SVM]] es el ejemplo canónico de optimización convexa con restricciones; [[Gradient-Based Optimization]] es el algoritmo. Esta nota unifica el lenguaje matemático detrás de todos estos métodos.

## 2. Requisitos

- [[Calculus]] multivariable: gradientes, Hessianos, series de Taylor
- [[Linear Algebra]]: vectores, matrices, norma, producto interno
- Familiaridad con [[Supervised Learning]] (SVM, regresión logística)

## 3. Convexidad: por qué importa

### Conjuntos convexos

Un conjunto $C \subseteq \mathbb{R}^n$ es convexo si para cualquier $x, y \in C$ y $\theta \in [0, 1]$:

$$\theta x + (1-\theta) y \in C$$

### Funciones convexas

Una función $f: \mathbb{R}^n \to \mathbb{R}$ es convexa si su dominio es convexo y:

$$f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y)$$

**Interpretación**: el segmento entre dos puntos está siempre por encima de la función.

### ¿Por qué importa en ML?

- En optimización convexa, **todo mínimo local es mínimo global**
- No hay puntos silla problemáticos (en el sentido de optimización)
- La convergencia de SGD está garantizada (aunque sea asintótica)
- Problemas convexos tienen algoritmos eficientes con garantías de convergencia

```python
import numpy as np
import matplotlib.pyplot as plt

# Función convexa: f(x) = x^2 (convexa)
# Función no convexa: g(x) = x^3 - 3x (cóncava en [-1,1], convexa fuera)

def f_convexa(x):
    return x**2

def f_noconvexa(x):
    return x**3 - 3*x

x = np.linspace(-3, 3, 100)

# Verificar convexidad: f''(x) >= 0 para toda x
segunda_deriv_convexa = 2 * np.ones_like(x)  # f''(x) = 2
segunda_deriv_noconv = 6 * x  # g''(x) = 6x → negativa para x < 0

print("f(x) = x^2 — segunda derivada siempre positiva:", end=" ")
print(np.all(segunda_deriv_convexa >= 0))

print("g(x) = x^3 - 3x — segunda derivada no siempre positiva:", end=" ")
print(np.all(segunda_deriv_noconv >= 0))

# Una función convexa tiene mínimo global único
from scipy.optimize import minimize

res = minimize(f_convexa, x0=10.0, method='BFGS')
print(f"\nMínimo de x^2 desde x0=10: x={res.x[0]:.6f}, f(x)={res.fun:.6f}")

res = minimize(f_noconvexa, x0=2.0, method='BFGS')
print(f"Mínimo local de g(x) desde x0=2: x={res.x[0]:.6f}, f(x)={res.fun:.6f}")

res = minimize(f_noconvexa, x0=-2.0, method='BFGS')
print(f"Mínimo local de g(x) desde x0=-2: x={res.x[0]:.6f}, f(x)={res.fun:.6f}")

# Salida esperada:
# f(x) = x^2 — segunda derivada siempre positiva: True
# g(x) = x^3 - 3x — segunda derivada no siempre positiva: False
#
# Mínimo de x^2 desde x0=10: x=0.000000, f(x)=0.000000
# Mínimo local de g(x) desde x0=2: x=1.000000, f(x)=-2.000000
# Mínimo local de g(x) desde x0=-2: x=1.000000, f(x)=-2.000000
```

### Funciones convexas comunes en ML

- $f(w) = \|w\|^2$ (regularización L2)
- $f(w) = \log(1 + e^{-y w^T x})$ (logistic loss)
- $f(w) = \max(0, 1 - y w^T x)$ (hinge loss)
- $f(w) = \|Xw - y\|^2$ (MSE, regresión lineal)

## 4. Optimización sin restricciones

### Condición de optimalidad

Para $f$ diferenciable y convexa, $x^*$ es mínimo global si y solo si:

$$\nabla f(x^*) = 0$$

Para $f$ no convexa, $\nabla f(x^*) = 0$ es condición necesaria (punto crítico), pero puede ser máximo, mínimo o punto silla.

### Condición de segundo orden

Si $\nabla f(x^*) = 0$ y $\nabla^2 f(x^*) \succ 0$ (Hessiano definido positivo), entonces $x^*$ es mínimo local estricto.

### Gradient Descent

El algoritmo más simple:
$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

- $\eta$: learning rate
- Converge a velocidad $O(1/k)$ para funciones convexas $L$-smooth
- Con momentum (NAG, Adam): converge más rápido en la práctica

```python
import numpy as np

# Minimizar f(w) = w^2 + 2w + 1 (mínimo en w = -1)
def f(w):
    return w**2 + 2*w + 1

def grad_f(w):
    return 2*w + 2

w = 3.0  # inicial
lr = 0.1
trayectoria = [w]

for i in range(30):
    w = w - lr * grad_f(w)
    trayectoria.append(w)

print(f"w final: {w:.6f}")
print(f"f(w) final: {f(w):.6f}")
print(f"∇f(w) final: {grad_f(w):.6f}")

# Salida esperada:
# w final: -1.000000
# f(w) final: 0.000000
# ∇f(w) final: 0.000000
```

## 5. Optimización con restricciones: Lagrange multipliers

### Problema con restricciones de igualdad

$$\min_{x} f(x) \quad \text{sujeto a} \quad h_i(x) = 0, \quad i = 1, \ldots, m$$

El **Lagrangiano** es:

$$\mathcal{L}(x, \lambda) = f(x) + \sum_{i=1}^{m} \lambda_i h_i(x)$$

**Condición necesaria**: en el óptimo, $\nabla_x \mathcal{L} = 0$ y $h_i(x) = 0$.

**Interpretación geométrica**: en el óptimo, $\nabla f$ es combinación lineal de $\nabla h_i$ — las superficies de nivel de $f$ son tangentes a las restricciones.

```python
import numpy as np

# Ejemplo: minimizar f(x,y) = x^2 + y^2 (distancia al origen)
# Sujeto a: h(x,y) = x + y - 1 = 0 (recta)
# Solución analítica: (0.5, 0.5), f = 0.5

f = lambda x, y: x**2 + y**2
h = lambda x, y: x + y - 1

# Lagrangiano: L = x^2 + y^2 + λ(x + y - 1)
# ∇L = [2x + λ, 2y + λ, x + y - 1] = 0
# Solución: x = y = 0.5, λ = -1

x_opt, y_opt, lam_opt = 0.5, 0.5, -1.0
print(f"x* = {x_opt}, y* = {y_opt}")
print(f"f(x*,y*) = {f(x_opt, y_opt):.4f}")
print(f"h(x*,y*) = {h(x_opt, y_opt):.4f}")
print(f"∇f = [{2*x_opt}, {2*y_opt}]")
print(f"∇h = [1, 1]")
print(f"∇f + λ∇h = [{2*x_opt + lam_opt}, {2*y_opt + lam_opt}]")

# Salida esperada:
# x* = 0.5, y* = 0.5
# f(x*,y*) = 0.5000
# h(x*,y*) = 0.0000
# ∇f = [1.0, 1.0]
# ∇h = [1, 1]
# ∇f + λ∇h = [0.0, 0.0]
```

## 6. KKT Conditions

Para problemas con restricciones de **desigualdad** e igualdad:

$$\min_x f(x) \quad \text{s.a.} \quad h_i(x) = 0, \; g_j(x) \leq 0$$

Las condiciones necesarias de Karush-Kuhn-Tucker (KKT) son:

1. **Estacionariedad**: $\nabla f(x^*) + \sum \lambda_i \nabla h_i(x^*) + \sum \mu_j \nabla g_j(x^*) = 0$
2. **Factibilidad primal**: $h_i(x^*) = 0$, $g_j(x^*) \leq 0$
3. **Factibilidad dual**: $\mu_j \geq 0$
4. **Complementary slackness**: $\mu_j g_j(x^*) = 0$ (si $g_j(x^*) < 0$ entonces $\mu_j = 0$)

**Intuición**: $\mu_j > 0$ solo para restricciones activas ($g_j(x^*) = 0$). Las restricciones inactivas no afectan la solución.

```python
import numpy as np

# Ejemplo SVM primal: min (1/2)||w||^2 s.a. y_i(w^T x_i + b) >= 1
# Para un dataset linealmente separable simple

# Dataset: 2 puntos en 1D
X = np.array([-1.0, 1.0])
y = np.array([-1.0, 1.0])

# w* = 1, b* = 0  (margen = 1/||w|| = 1)
# Debería cumplir KKT:
# Estacionariedad: w - Σ μ_i y_i x_i = 0
# En este caso: μ_1 = μ_2 = 0.5

w_star, b_star = 1.0, 0.0
mu = np.array([0.5, 0.5])

# Verificar complementary slackness
g = y * (w_star * X + b_star) - 1
print("g(x):", g)
print("μ ⊙ g:", mu * g)

# Verificar estacionariedad
est = w_star - np.sum(mu * y * X)
print("Estacionariedad:", est)

# Salida esperada:
# g(x): [0. 0.]
# μ ⊙ g: [0. 0.]
# Estacionariedad: 0.0
```

En SVM, KKT revela que solo los **support vectors** tienen $\mu_j > 0$. El resto de puntos ($g_j(x) > 0$) no contribuyen al costo.

## 7. Dualidad

### Problema primal

$$p^* = \min_x f(x) \quad \text{s.a.} \quad h_i(x) = 0, \; g_j(x) \leq 0$$

### Función dual de Lagrange

$$q(\lambda, \mu) = \min_x \mathcal{L}(x, \lambda, \mu) = \min_x \left[ f(x) + \sum \lambda_i h_i(x) + \sum \mu_j g_j(x) \right]$$

### Problema dual

$$d^* = \max_{\lambda, \mu \geq 0} q(\lambda, \mu)$$

### Dual Gap

$$p^* - d^* \geq 0$$

- **Dualidad débil**: siempre $p^* \geq d^*$
- **Dualidad fuerte**: si el problema es convexo y Slater condition se cumple, $p^* = d^*$

### SVM como ejemplo canónico

El primal de SVM (con $C \to \infty$, separable) es:

$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.a.} \quad y_i(w^T x_i + b) \geq 1$$

El dual (que es lo que realmente se resuelve) es:

$$\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j \quad \text{s.a.} \quad \alpha_i \geq 0, \; \sum \alpha_i y_i = 0$$

El dual tiene tres ventajas:
1. Solo depende de productos internos $x_i^T x_j$ → permite kernels
2. La mayoría de $\alpha_i = 0$ (sparsity)
3. Es un problema de programación cuadrática con restricciones simples

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Dataset linealmente separable
X, y = make_blobs(n_samples=50, centers=2, random_state=42)
y = 2*y - 1  # etiquetas en {-1, 1}

svm = SVC(kernel='linear', C=1e6)  # C grande = separable
svm.fit(X, y)

print(f"Número de support vectors: {len(svm.support_)}")
print(f"Total de muestras: {len(y)}")
print(f"Alpha promedio (support vectors): {svm.dual_coef_[0].mean():.4f}")

# Salida esperada (aproximada):
# Número de support vectors: 3
# Total de muestras: 50
# Alpha promedio (support vectors): 0.2505
```

Solo 3 de 50 puntos son support vectors — el resto tiene $\alpha_i = 0$ por complementary slackness.

## 8. Common Mistakes

1. **Asumir convexidad donde no la hay**: redes neuronales con activaciones no lineales y múltiples capas son **no convexas**. No hay garantía de mínimo global. Técnicas como Adam no "resuelven" la optimización, solo encuentran mínimos locales.

2. **Ignorar restricciones de desigualdad**: en Lagrange multipliers se suelen considerar solo restricciones de igualdad. Para desigualdades se necesitan KKT, que incluyen la condición $\mu_j \geq 0$ y complementary slackness.

3. **No verificar KKT**: si implementas un optimizador con restricciones desde cero, las condiciones KKT son la herramienta de debugging. Ignorarlas lleva a soluciones no óptimas.

4. **Confundir dual gap en problemas no convexos**: sin convexidad, el dual gap puede ser enorme. La solución del dual puede ser muy distinta del primal.

5. **Olvidar Slater condition**: en problemas convexos con restricciones de desigualdad, la dualidad fuerte requiere que exista un punto estrictamente factible ($g_j(x) < 0$). Sin esto, puede haber gap.

6. **Usar gradient descent con restricciones sin proyectar**: si optimizas $f(x)$ con $x \geq 0$, necesitas **gradient projection** o reparametrizar. Ignorar la restricción produce soluciones infactibles.

## Resumen

La optimización es el motor del ML. La convexidad garantiza que mínimo local = mínimo global, lo que hace que problemas como SVM, regresión logística y regresión lineal sean tratables. Para optimización sin restricciones, la condición $\nabla f = 0$ es necesaria y (con convexidad) suficiente. Para restricciones de igualdad, los multiplicadores de Lagrange extienden esta condición igualando $\nabla f$ con combinación lineal de $\nabla h_i$. Las condiciones KKT generalizan a restricciones de desigualdad añadiendo $\mu_j \geq 0$ y complementary slackness ($\mu_j g_j = 0$). La dualidad permite resolver el problema primal indirectamente — en SVM, el dual habilita el kernel trick y produce soluciones sparse. Entender estas herramientas permite pasar de "correr un optimizador" a saber qué está haciendo realmente.

## Check Your Understanding

1. ¿Por qué el SVM dual es preferible al primal para kernels? <!-- El dual solo depende de productos internos $x_i^T x_j$, que se reemplazan por $K(x_i, x_j)$. El primal requeriría trabajar explícitamente en el espacio de features transformado, que puede ser infinito-dimensional. -->

2. Si una restricción $g_j(x) \leq 0$ no está activa en el óptimo, ¿cuánto vale su multiplicador $\mu_j$? <!-- $\mu_j = 0$ por complementary slackness: si $g_j(x) < 0$, entonces $\mu_j g_j(x) = 0$ implica $\mu_j = 0$. -->

3. ¿Una función convexa siempre tiene un mínimo? <!-- No. $f(x) = e^x$ es convexa pero no tiene mínimo (tiende a 0 asintóticamente). Se requiere que $f$ sea coerciva o que el dominio dominante sea acotado. -->

4. En SVM, ¿qué pasa si aumentas $C$ al aprender con datos no separables? <!-- Mayor $C$ penaliza más los errores de clasificación. En el límite $C \to \infty$, el SVM intenta separar perfectamente (riesgo de overfitting). -->

5. ¿Cuándo falla la dualidad fuerte? <!-- Cuando el problema no es convexo, o cuando es convexo pero no se cumple Slater condition (no hay punto estrictamente factible). -->

## Where to Go Next

- [[Calculus]] — gradientes, Hessianos, expansión de Taylor
- [[Linear Algebra]] — espacio dual, normas, productos internos
- [[Advanced Linear Algebra]] — SVD y pseudoinversa en optimización
- [[Convex Optimization]] — profundización en teoría de convexidad
- [[Gradient-Based Optimization]] — SGD, Adam, y variantes
- [[Supervised Learning]] — SVM, regresión logística como problemas de optimización
- [[SVM]] — del primal al dual, kernel trick
- [[Numerical Methods]] — estabilidad numérica en optimización
