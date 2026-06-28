---
tags: [mathematics, calculus, foundational]
status: growing
created: 2026-06-27
---

# Calculus

## 1. Why This Matters

Machine learning is **[[Gradient-Based Optimization|optimization]]**. Every model you train — from linear regression to GPT — is solving an optimization problem: find the parameters that minimize the error (the loss function). And the primary tool for optimization is the **derivative**.

Gradient descent, which is the algorithm that trains almost every neural network, is nothing more than repeatedly taking small steps in the direction opposite to the derivative. Backpropagation, which computes how each weight contributed to the error, is just repeated application of the **chain rule** from calculus.

Without calculus, there is no learning. You can train models with scikit-learn without thinking about derivatives, but the moment you want to understand why a learning rate of 0.01 works better than 0.1, or why Adam converges faster than SGD, or why your loss exploded after layer 50 — you need calculus.

---

## 2. Derivatives — The Rate of Change

### 2.1 Intuition

The derivative tells you **how fast something is changing** at a specific point. If you are driving and your speedometer reads 60 km/h, that is a derivative: the rate of change of your position with respect to time.

In ML, the derivative of the loss with respect to a weight tells you: "if I increase this weight by a tiny amount, does the loss go up or down, and by how much?"

Formally:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

**Visual description**: draw the curve $f(x)$. At point $x$, draw a tangent line. The slope of that line is $f'(x)$. Positive slope → increasing. Negative slope → decreasing. Steeper slope → faster change.

### 2.2 Key Derivative Rules

**Power rule**: $\frac{d}{dx} x^n = n x^{n-1}$
- Example: $\frac{d}{dx} x^2 = 2x$, $\frac{d}{dx} x^3 = 3x^2$

**Exponential**: $\frac{d}{dx} e^x = e^x$
- This is special: the exponential function is its own derivative
- It is why the Sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ has a simple derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**Logarithm**: $\frac{d}{dx} \ln x = \frac{1}{x}$
- Used in cross-entropy loss calculations

**Chain rule**: $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$
- This is **the fundamental rule for deep learning**

### 2.3 Partial Derivatives

When a function has multiple inputs (like a neural network with many weights), you need **partial derivatives**. The partial derivative $\frac{\partial f}{\partial x_i}$ measures how $f$ changes when only $x_i$ changes, keeping everything else fixed.

**Example**: $f(x, y) = x^2 y + y^3$
- $\frac{\partial f}{\partial x} = 2xy$
- $\frac{\partial f}{\partial y} = x^2 + 3y^2$

### 2.4 The Gradient

The **gradient** $\nabla f$ is a vector of all partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & ... & \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Intuition**: the gradient points in the direction of **steepest ascent**. If you want to increase $f$ as fast as possible, follow the gradient. If you want to decrease $f$ (minimize the loss), follow the **negative gradient**.

This is literally what gradient descent does:

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

---

## 3. Gradient Descent — The Learning Algorithm

### 3.1 How It Works

Imagine you are blindfolded on a mountain and want to reach the valley below. You feel the ground with your foot to find which direction goes downhill (the gradient), take a step in that direction, and repeat. The size of your step is the **learning rate**.

**Algorithm**:
1. Start with random weights $w$
2. Compute the loss $L(w)$ over your training data
3. Compute the gradient $\nabla L(w)$ — direction of steepest increase
4. Update: $w = w - \eta \nabla L(w)$ (step opposite to gradient)
5. Repeat until the loss stops decreasing

### 3.2 The Learning Rate $\eta$

This is the most important hyperparameter.

- **Too large**: you overshoot the minimum. The loss may even diverge (explode to infinity).
- **Too small**: you make painfully slow progress. Training takes forever.
- **Just right**: you converge efficiently.

In practice, learning rates typically range from $10^{-6}$ to $10^{-1}$, depending on the model and task.

### 3.3 Stochastic Gradient Descent (SGD)

Computing the gradient over ALL training data (full batch) is expensive when you have millions of examples. SGD uses a **mini-batch** (e.g., 32 or 256 samples) to estimate the gradient.

Why this works: the gradient over a random mini-batch is an **unbiased estimate** of the true gradient. It is noisy, but each step is much cheaper, so overall progress is faster.

### 3.4 Variants

| Optimizer | Key Idea |
|---|---|
| **SGD + Momentum** | Accumulate past gradient directions to smooth updates and escape local minima |
| **Adam** | Adaptive learning rate per parameter + momentum. Most common default. |
| **AdamW** | Adam with decoupled weight decay. Better for transformers. |
| **RMSprop** | Adaptive learning rate. Works well for RNNs. |

---

## 4. Backpropagation — The Chain Rule in Action

### 4.1 The Problem

A neural network is a composition of many functions:

$$y = f_4(f_3(f_2(f_1(x))))$$

To train it, we need the gradient of the loss with respect to **every weight** in every layer. For a 50-layer network with millions of weights, that seems impossible.

### 4.2 The Solution — Chain Rule

The chain rule lets us decompose the gradient of a composition into a product of simpler gradients:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial w_1}$$

**Concrete example** with a 2-layer network:

```
x → (W₁, b₁) → h₁ → σ(h₁) → (W₂, b₂) → ŷ → Loss(y, ŷ)
```

To update $W_1$, we compute:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

Backpropagation computes this efficiently by:
1. **Forward pass**: compute all activations (store them)
2. **Backward pass**: compute gradients from the last layer backwards, reusing previously computed gradients

### 4.3 Vanishing and Exploding Gradients

**Vanishing gradients**: when the chain rule multiplies many small numbers (< 1), the gradient becomes exponentially smaller as you go back through layers. Early layers barely learn.

- Cause: sigmoid/tanh activations (derivative max = 0.25). After 50 layers: $0.25^{50} \approx 10^{-30}$
- Fix: ReLU activation, residual connections (ResNet), proper initialization

**Exploding gradients**: when gradients become exponentially larger in early layers.

- Cause: poor initialization, deep networks without normalization
- Fix: gradient clipping (cap the gradient norm), better initialization (He, Xavier), batch normalization

---

## 5. Integrals — A Brief Note

You will not compute integrals directly in ML nearly as often as derivatives, but they appear in:

- **Expected value**: $E[X] = \int x \cdot p(x) dx$
- **Marginalization**: $p(x) = \int p(x, y) dy$
- **KL divergence**: measures difference between two probability distributions
- **[[Bayesian Inference]]**: computing posterior probabilities often involves intractable integrals, which is why we use approximations (MCMC, variational inference)

The key intuition: an integral is the **area under a curve**. If the curve is a probability distribution, the integral over a range gives the probability of that range.

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

## 8. Summary

Derivatives tell you how to change your model to improve it. Gradient descent is the algorithm that takes those derivatives and turns them into better weights. Backpropagation computes derivatives efficiently for millions of parameters. Every optimizer (SGD, Adam, etc.) is a refinement of the same core idea: follow the negative gradient.

---

## 9. Where to Go Next

- [[Linear Algebra]] — Gradients are vectors; transforms in weight space
- [[Neural Networks]] — Backpropagation trains every layer
- [[Training Techniques]] — Adam, learning rate schedules, gradient clipping
- [[Probability]] — Maximum likelihood estimation uses derivatives
