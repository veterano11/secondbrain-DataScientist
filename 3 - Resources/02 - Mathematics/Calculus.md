---
tags: [mathematics, calculus, foundational]
status: growing
created: 2026-06-27
---

# Cálculo

## 1. Escenario de aprendizaje

Estás entrenando una red neuronal profunda y ajustas la tasa de aprendizaje: 0.1 hace que la pérdida explote a infinito, 0.001 hace que el entrenamiento avance tan lento que nunca termina, pero 0.01 funciona perfectamente. ¿Por qué? La respuesta está en el cálculo. El descenso por gradiente, que es el algoritmo que entrena casi todas las redes neuronales, no es más que repetir pequeños pasos en la dirección opuesta a la derivada. La retropropagación, que calcula cómo cada peso contribuyó al error, es solo la aplicación repetida de la **regla de la cadena** del cálculo.

El machine learning es **optimización**. Cada modelo que entrenas — desde regresión lineal hasta GPT — resuelve un problema de optimización: encontrar los parámetros que minimizan el error (la función de pérdida). Y la herramienta principal para la optimización es la **derivada**. Sin cálculo, no hay aprendizaje.

---

## 2. Derivadas — La tasa de cambio

### 2.1 Intuición

La derivada te dice **qué tan rápido está cambiando algo** en un punto específico. Si estás conduciendo y tu velocímetro marca 60 km/h, eso es una derivada: la tasa de cambio de tu posición con respecto al tiempo.

En ML, la derivada de la pérdida con respecto a un peso te dice: "si aumento este peso una cantidad muy pequeña, ¿la pérdida sube o baja, y en qué medida?"

Formalmente:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

**Descripción visual**: dibuja la curva $f(x)$. En el punto $x$, traza una recta tangente. La pendiente de esa recta es $f'(x)$. Pendiente positiva → creciente. Pendiente negativa → decreciente. Pendiente más pronunciada → cambio más rápido.

### 2.2 Reglas de derivación clave

**Regla de la potencia**: $\frac{d}{dx} x^n = n x^{n-1}$
- Ejemplo: $\frac{d}{dx} x^2 = 2x$, $\frac{d}{dx} x^3 = 3x^2$

**Exponencial**: $\frac{d}{dx} e^x = e^x$
- Esto es especial: la función exponencial es su propia derivada
- Es por eso que la función Sigmoide $\sigma(x) = \frac{1}{1 + e^{-x}}$ tiene una derivada simple: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**Logaritmo**: $\frac{d}{dx} \ln x = \frac{1}{x}$
- Se usa en cálculos de pérdida de entropía cruzada (cross-entropy)

**Regla de la cadena**: $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$
- Esta es **la regla fundamental para el deep learning**

### 2.3 Derivadas parciales

Cuando una función tiene múltiples entradas (como una red neuronal con muchos pesos), necesitas **derivadas parciales**. La derivada parcial $\frac{\partial f}{\partial x_i}$ mide cómo cambia $f$ cuando solo cambia $x_i$, manteniendo todo lo demás fijo.

**Ejemplo**: $f(x, y) = x^2 y + y^3$
- $\frac{\partial f}{\partial x} = 2xy$
- $\frac{\partial f}{\partial y} = x^2 + 3y^2$

### 2.4 El Gradiente

El **gradiente** $\nabla f$ es un vector de todas las derivadas parciales:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & ... & \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Intuición**: el gradiente apunta en la dirección de **máximo ascenso**. Si quieres aumentar $f$ lo más rápido posible, sigue el gradiente. Si quieres disminuir $f$ (minimizar la pérdida), sigue el **gradiente negativo**.

Esto es literalmente lo que hace el descenso por gradiente:

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

---

## 3. Descenso por Gradiente — El algoritmo de aprendizaje

### 3.1 Cómo funciona

Imagina que estás con los ojos vendados en una montaña y quieres llegar al valle. Sientes el suelo con el pie para encontrar qué dirección va cuesta abajo (el gradiente), das un paso en esa dirección y repites. El tamaño de tu paso es la **tasa de aprendizaje**.

**Algoritmo**:
1. Comienza con pesos aleatorios $w$
2. Calcula la pérdida $L(w)$ sobre tus datos de entrenamiento
3. Calcula el gradiente $\nabla L(w)$ — dirección de máximo incremento
4. Actualiza: $w = w - \eta \nabla L(w)$ (paso opuesto al gradiente)
5. Repite hasta que la pérdida deje de disminuir

### 3.2 La tasa de aprendizaje $\eta$

Este es el hiperparámetro más importante.

- **Demasiado grande**: te pasas del mínimo. La pérdida puede incluso divergir (explotar a infinito).
- **Demasiado pequeño**: avanzas dolorosamente lento. El entrenamiento toma una eternidad.
- **Justo el correcto**: converves eficientemente.

En la práctica, las tasas de aprendizaje típicamente van de $10^{-6}$ a $10^{-1}$, dependiendo del modelo y la tarea.

### 3.3 Descenso por Gradiente Estocástico (SGD)

Calcular el gradiente sobre TODOS los datos de entrenamiento (batch completo) es costoso cuando tienes millones de ejemplos. SGD usa un **mini-batch** (ej. 32 o 256 muestras) para estimar el gradiente.

Por qué funciona: el gradiente sobre un mini-batch aleatorio es una **estimación insesgada** del gradiente verdadero. Es ruidoso, pero cada paso es mucho más barato, por lo que el progreso general es más rápido.

### 3.4 Variantes

| Optimizador | Idea clave |
|---|---|
| **SGD + Momentum** | Acumula direcciones de gradiente pasadas para suavizar las actualizaciones y escapar de mínimos locales |
| **Adam** | Tasa de aprendizaje adaptativa por parámetro + momentum. El valor predeterminado más común. |
| **AdamW** | Adam con decaimiento de peso desacoplado. Mejor para transformers. |
| **RMSprop** | Tasa de aprendizaje adaptativa. Funciona bien para RNNs. |

---

## 4. Retropropagación — La regla de la cadena en acción

### 4.1 El problema

Una red neuronal es una composición de muchas funciones:

$$y = f_4(f_3(f_2(f_1(x))))$$

Para entrenarla, necesitamos el gradiente de la pérdida con respecto a **cada peso** en cada capa. Para una red de 50 capas con millones de pesos, eso parece imposible.

### 4.2 La solución — Regla de la cadena

La regla de la cadena nos permite descomponer el gradiente de una composición en un producto de gradientes más simples:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial w_1}$$

**Ejemplo concreto** con una red de 2 capas:

```
x → (W₁, b₁) → h₁ → σ(h₁) → (W₂, b₂) → ŷ → Loss(y, ŷ)
```

Para actualizar $W_1$, calculamos:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

La retropropagación calcula esto eficientemente mediante:
1. **Pase hacia adelante (forward pass)**: calcula todas las activaciones (almacénalas)
2. **Pase hacia atrás (backward pass)**: calcula los gradientes desde la última capa hacia atrás, reutilizando gradientes previamente calculados

### 4.3 Gradientes vanishing y exploding

**Gradientes vanishing (desvanecientes)**: cuando la regla de la cadena multiplica muchos números pequeños (< 1), el gradiente se vuelve exponencialmente más pequeño a medida que retrocedes por las capas. Las primeras capas apenas aprenden.

- Causa: activaciones sigmoid/tanh (derivada máxima = 0.25). Después de 50 capas: $0.25^{50} \approx 10^{-30}$
- Solución: activación ReLU, conexiones residuales (ResNet), inicialización adecuada

**Gradientes exploding (explosivos)**: cuando los gradientes se vuelven exponencialmente más grandes en las primeras capas.

- Causa: mala inicialización, redes profundas sin normalización
- Solución: recorte de gradiente (limitar la norma del gradiente), mejor inicialización (He, Xavier), batch normalization

---

## 5. Integrales — Una breve nota

No calcularás integrales directamente en ML tan a menudo como derivadas, pero aparecen en:

- **Valor esperado**: $E[X] = \int x \cdot p(x) dx$
- **Marginalización**: $p(x) = \int p(x, y) dy$
- **Divergencia KL**: mide la diferencia entre dos distribuciones de probabilidad
- **[[Bayesian Inference]]**: calcular probabilidades posteriores a menudo implica integrales intratables, por lo que usamos aproximaciones (MCMC, inferencia variacional)

La intuición clave: una integral es el **área bajo una curva**. Si la curva es una distribución de probabilidad, la integral sobre un rango da la probabilidad de ese rango.

---

## 6. Common Mistakes

1. **Confusing local and global minima**: gradient descent finds a local minimum. In [[Convex Optimization|convex problems]] (linear regression), the local minimum is global. In deep learning, many local minima have similar loss values.

2. **Setting the learning rate too high**: the most common training mistake. If your loss oscillates or goes to NaN, reduce the learning rate.

3. **Not normalizing inputs**: if features have different scales, the loss surface is elongated and gradient descent zigzags. Normalize to help optimization.

4. **Thinking backpropagation is complex**: It is just the chain rule applied efficiently. The "magic" is in the bookkeeping, not the math.

5. **Ignoring the gradient flow**: if early layers are not learning (weights barely change), check for vanishing gradients. If training is unstable, check for exploding gradients.

---

## 7. Check Your Understanding

1. What is the derivative of the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$? Why is it convenient for neural networks?
2. If the learning rate is too high, what happens to the loss curve during training? Draw it mentally.
3. Why does SGD with momentum converge faster than plain SGD?
4. A 100-layer network with ReLU activations still suffers from vanishing gradients. What might be the cause?
5. Why is the derivative of $L_2$ regularization $\lambda\|w\|^2$ equal to $2\lambda w$? What does this mean for the weight update?

---

## 8. Resumen

Derivatives tell you how to change your model to improve it. Gradient descent is the algorithm that takes those derivatives and turns them into better weights. Backpropagation computes derivatives efficiently for millions of parameters. Every optimizer (SGD, Adam, etc.) is a refinement of the same core idea: follow the negative gradient.

---

## 9. Where to Go Next

- [[Linear Algebra]] — Gradients are vectors; transforms in weight space
- [[Neural Networks]] — Backpropagation trains every layer
- [[Training Techniques]] — Adam, learning rate schedules, gradient clipping
- [[Probability]] — Maximum likelihood estimation uses derivatives
