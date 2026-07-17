---
tags: [optimization, convex, mathematics]
status: growing
created: 2026-06-27
---

# Convex Optimization

## 1. Escenario de aprendizaje

Entrenás una regresión lineal con Ridge. Sin importar los datos de inicialización, siempre obtenés los mismos coeficientes. Eso es porque Ridge es un problema de optimización **convexo** — tiene un único mínimo global.

En cambio, una red neuronal puede dar resultados distintos en cada entrenamiento porque es **no convexa** — tiene múltiples mínimos locales.

Entender la convexidad te dice: ¿puedo encontrar el óptimo global garantizado, o tengo que conformarme con un mínimo local?

## 2. ¿Qué es un convexo?

### 2.1 Conjunto convexo

Un conjunto $C$ es convexo si el segmento entre cualquier par de puntos está dentro de $C$:

$$\theta x + (1-\theta) y \in C \quad \forall x, y \in C, \theta \in [0, 1]$$

**Ejemplos**: un círculo es convexo. Una medialuna no (el segmento entre dos puntos cruza fuera).

### 2.2 Función convexa

Una función $f$ es convexa si el segmento entre dos puntos siempre está por encima de la función:

$$f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y)$$

**Test práctico**: $\nabla^2 f(x) \succeq 0$ (Hessiana semidefinida positiva — todos los autovalores $\geq 0$). El gradiente y la Hessiana vienen del [[Calculus]].

| Función | ¿Convexa? | Hessiana |
|---------|-----------|----------|
| $f(x) = x^2$ | Sí | $2 > 0$ |
| $f(x) = x^4$ | Sí | $12x^2 \geq 0$ |
| $f(x) = \sin(x)$ | No | $-\sin(x)$ cambia signo |
| $f(x) = \|x\|$ | Sí | No existe en 0, pero es convexa |

## 3. Propiedad clave: mínimo global garantizado

En optimización convexa, **todo mínimo local es mínimo global**.

Para problemas no convexos (deep learning), esto no vale. Pero muchos problemas de [[Supervised Learning|ML]] son convexos:

| Modelo | ¿Convexo? |
|--------|-----------|
| Linear regression (MSE) | Sí |
| Ridge regression | Sí |
| Lasso | Sí (convexa pero no diferenciable en 0) |
| Logistic regression | Sí |
| SVM (hinge loss) | Sí |
| Neural networks | **No** |

## 4. Problema de optimización convexa

Minimizar $f_0(x)$ sujeto a $f_i(x) \leq 0$, $h_j(x) = 0$ (con $f_i$ convexas, $h_j$ afines).

### 4.1 Ejemplo típico: Lasso

$$\min_w \|Xw - y\|_2^2 + \lambda\|w\|_1$$

Ambos términos son convexos. La suma también. Mínimo global garantizado.

```python
import cvxpy as cp

w = cp.Variable(n)
lambda_reg = 0.1

objective = cp.Minimize(cp.norm(X @ w - y, 2) + lambda_reg * cp.norm(w, 1))
problem = cp.Problem(objective)
problem.solve()

print(f"Solución: {w.value}")
```

### 4.2 LP (Linear Programming)

$$ \min c^T x \quad \text{s.a. } Ax \leq b, x \geq 0 $$

```python
c = [-5, -3]      # maximizar 5x + 3y
A = [[2, 1], [1, 1]]
b = [40, 30]

from scipy.optimize import linprog
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
print(f"Óptimo: {res.fun}, x={res.x}")
```

## 5. Dualidad

Cada problema convexo (primal) tiene un problema **dual**. El dual da una cota inferior de la solución óptima.

**Lagrangiano**: $L(x, \lambda, \nu) = f_0(x) + \sum \lambda_i f_i(x) + \sum \nu_j h_j(x)$

**Función dual**: $g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$ — siempre cóncava.

**Dualidad fuerte**: si se cumple Slater's condition, el óptimo primal = óptimo dual.

**KKT conditions**: condiciones necesarias y suficientes para optimalidad en convexos.

## 6. ¿Puedo formular mi problema como convexo?

| Problema ML | ¿Formulación convexa? |
|-------------|----------------------|
| Regresión | Sí |
| Clasificación (logistic, SVM) | Sí |
| Feature selection (Lasso) | Sí |
| Neural networks | No |
| PCA | No (Stiefel manifold) |
| Clustering (k-means) | No (NP-hard) |

Si tu problema no es convexo, el buen punto de partida es encontrar una **relajación convexa** — un problema convexo que da una cota o una inicialización.

## 7. Common Mistakes

1. **Asumir que todo mínimo local es global**: solo para convexos. En deep learning, los mínimos locales son raros; los **saddle points** son el verdadero problema.
2. **Ignorar condiciones KKT**: para problemas convexos con restricciones, KKT es necesaria y suficiente. Si no se cumplen, la solución no es óptima.
3. **Resolver con [[Training Techniques|SGD]] un problema que tiene solución cerrada**: para Ridge (convexo), $(X^T X + \lambda I)^{-1} X^T y$ es directo. No necesitás iteraciones.

## 8. Check Your Understanding

1. ¿$f(x) = e^x$ es convexa? ¿Y $f(x) = \log(x)$?
2. ¿Por qué es importante saber si un problema es convexo antes de elegir un solver?
3. En SVM, el objetivo es convexo. ¿Qué garantía te da esto sobre el entrenamiento?

**Respuestas rápidas:**
1. $e^x$: sí ($e^x > 0$). $\log(x)$: no ($-1/x^2 < 0$), es cóncava.
2. Si es convexo, podés usar solvers especializados que garantizan el óptimo global rápidamente. Si no, necesitás métodos heurísticos (SGD, annealing).
3. Que SGD va a converger al clasificador de máximo margen, sin importar la inicialización.

## 9. Resumen

La convexidad garantiza que el mínimo local es global y que podemos encontrarlo eficientemente. Muchos problemas clásicos de ML son convexos (regresión, SVM, logistic regression). Para problemas no convexos (deep learning), debemos aceptar soluciones aproximadas. La dualidad y las condiciones KKT son las herramientas para analizar y resolver problemas convexos.

## 10. Where to Go Next

- [[Gradient-Based Optimization]] — métodos iterativos para convexos y no convexos
- [[Hyperparameter Tuning]] — Bayesian optimization (no convexo en sí mismo)
- [[Linear Algebra]] — autovalores y matrices definidas positivas
- [[Regularization]] — Ridge y Lasso como problemas convexos
