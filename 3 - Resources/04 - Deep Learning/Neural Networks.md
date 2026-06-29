---
tags: [deep-learning, neural-networks, advanced]
status: growing
created: 2026-06-27
---

# Neural Networks

## 1. Escenario de aprendizaje

Quieres construir un sistema que reconozca dígitos escritos a mano (como los códigos postales en sobres). Tienes imágenes de 28×28 píxeles y quieres que la máquina aprenda a clasificarlas en dígitos del 0 al 9. Las redes neuronales son el motor detrás de esta tarea — desde reconocimiento de imágenes hasta modelos de lenguaje. La idea fundamental es simple: una red neuronal es una **composición de funciones diferenciables** que puede aproximar cualquier función continua (Teorema de Aproximación Universal). Pero la profundidad de la teoría está en entender cómo se componen esas funciones, cómo fluyen los gradientes a través de ellas y cómo la arquitectura afecta lo que la red puede aprender.

---

## 2. The Perceptron — The Building Block

### 2.1 The Neuron

A single neuron computes:

$$y = \sigma(w^T x + b) = \sigma\left(\sum_{i=1}^d w_i x_i + b\right)$$

**Step by step**:
1. Each input $x_i$ is multiplied by its weight $w_i$
2. All weighted inputs are summed with the bias $b$
3. The sum is passed through an activation function $\sigma$

**Analogy**: a neuron is a decision-making unit. Each input is evidence for or against some conclusion ($w_i$ tells you how important that evidence is). The bias is the baseline tendency. The activation function converts the total evidence into a decision.

### 2.2 Why Non-Linearity?

If we stack linear layers without activation functions, the entire network reduces to a single linear transformation:

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W'x + b'$$

A multi-layer linear network is no more powerful than a single layer. **Activation functions introduce non-linearity**, allowing the network to learn complex, non-linear relationships.

---

## 3. Activation Functions

### 3.1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

- **Historical importance**: the first widely used activation
- **Problem 1 — vanishing gradient**: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ has a maximum of 0.25. In deep networks, multiplying many small gradients causes the signal to vanish.
- **Problem 2 — not zero-centered**: outputs are always positive, causing zigzagging gradients
- **Still used for**: binary classification output layer (output as probability)

### 3.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$$

- Zero-centered (fixes sigmoid's zigzag problem)
- Still suffers from vanishing gradient (saturation at ±1)

### 3.3 ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- **The default activation** for hidden layers
- **Why it works**: derivative is 1 for $x > 0$ — no vanishing gradient!
- **Problem**: "dead ReLU" — if $x < 0$, the gradient is 0 and the neuron never activates
- **Fixes**: Leaky ReLU ($\alpha x$ for $x < 0$), PReLU (learnable $\alpha$)

### 3.4 GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi$ is the CDF of the standard normal distribution.

- Smoother than ReLU, used in modern transformers (GPT, BERT, Llama)
- Approximated as: $\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$

### 3.5 Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}$$

Converts a vector of $K$ real numbers into a probability distribution (all positive, sums to 1). Used in the **output layer for multi-class classification**.

---

## 4. Forward Propagation

Data flows through the network layer by layer:

$$h_0 = x \quad \text{(input)}$$
$$z_l = W_l h_{l-1} + b_l \quad \text{(linear transform)}$$
$$h_l = \sigma(z_l) \quad \text{(activation)}$$
$$\hat{y} = h_L \quad \text{(output)}$$

Each layer is a linear transformation followed by a non-linear activation. The composition of many such layers allows the network to learn hierarchical representations — simple features in early layers, complex features in later layers.

---

## 5. Backpropagation

### 5.1 The Problem

We have a network with millions of parameters. We need the gradient of the loss with respect to **each** parameter. Computing each derivative independently is impossible.

### 5.2 The Solution — Chain Rule

Backpropagation applies the chain rule efficiently:

1. **Forward pass**: compute and store all activations
2. **Backward pass**: compute gradients from the last layer backward, reusing previously computed values

**Concrete example** for a 2-layer network:

$$L = \frac{1}{2}(y - \hat{y})^2, \quad \hat{y} = W_2 \sigma(W_1 x + b_1) + b_2$$

1. Compute $\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$
2. Compute $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2} = (\hat{y} - y) \cdot h_1^T$
3. Compute $\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial \hat{y}} \cdot W_2^T$
4. Compute $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial (W_1 x)} \cdot x^T = \frac{\partial L}{\partial h_1} \odot \sigma'(z_1) \cdot x^T$

The key insight: the gradient at layer $l$ depends on the gradient at layer $l+1$. By computing backwards, each layer's gradient is reused.

### 5.3 Automatic Differentiation

Frameworks like PyTorch and TensorFlow implement backpropagation automatically (see [[Gradient-Based Optimization]] for more on gradient computation):

```python
import torch

x = torch.randn(10, requires_grad=True)
y = (x ** 2).sum()
y.backward()                    # computes all gradients
print(x.grad)                   # d(y)/dx = 2x
```

### 5.4 Vanishing and Exploding Gradients

**Vanishing**: in deep networks with sigmoid/tanh, gradients become exponentially smaller as you go back. Early layers learn extremely slowly or not at all.

- Fix: ReLU, residual connections, batch normalization, proper initialization

**Exploding**: gradients become exponentially larger in early layers, causing instability (NaN loss).

- Fix: gradient clipping, better initialization, lower learning rate

---

## 6. Loss Functions

| Task | Loss Function | Formula | Why This One |
|---|---|---|---|
| **Regression** | MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Differentiable, convex, penalizes large errors |
| **Regression** | MAE | $\frac{1}{n}\sum|y - \hat{y}|$ | Robust to outliers |
| **Binary class.** | BCE | $-\frac{1}{n}\sum y\log(\hat{y}) + (1-y)\log(1-\hat{y})$ | Derived from MLE for Bernoulli |
| **Multi-class** | Cross-entropy | $-\frac{1}{n}\sum\sum y_c\log(\hat{y}_c)$ | Derived from MLE for Multinomial |

**Why cross-entropy for classification?** It is equivalent to maximizing the likelihood of the predictions under the true distribution (a core [[Probability]] / MLE concept). Minimizing cross-entropy = maximizing the probability of the correct class.

---

## 7. Weight Initialization

Proper initialization prevents vanishing/exploding gradients at the start of training:

| Initialization | Distribution | Best for |
|---|---|---|
| **Xavier/Glorot** | $\text{Var}(w) = \frac{1}{\text{fan\_in}}$ | Tanh, Sigmoid |
| **He/Kaiming** | $\text{Var}(w) = \frac{2}{\text{fan\_in}}$ | ReLU |
| **Orthogonal** | Random orthogonal matrix | RNNs |

**Why it matters**: if weights are too large, activations explode. If too small, activations vanish. Correct initialization keeps activations in a reasonable range for all layers.

In modern frameworks, default initializations are usually good enough — but if training is unstable, initialization is worth checking.

---

## 8. Architecture Design Principles

- **Width vs depth**: deeper networks learn more abstract features, but are harder to train. Wider networks learn more patterns per layer but use more parameters.
- **Skip connections**: allow gradients to flow directly to early layers (ResNet). Essential for networks with 50+ layers.
- **Normalization**: batch norm, layer norm, or group norm stabilize training.
- **[[Regularization]]**: dropout, weight decay, early stopping prevent overfitting.

---

## 9. Common Mistakes

1. **ReLU in the output layer**: ReLU outputs $[0, \infty)$, which is meaningless as a probability or unbounded regression value. Use linear (regression), sigmoid (binary), or softmax (multi-class).

2. **No normalization**: deep networks without normalization are unstable. Use batch norm or layer norm.

3. **Too high learning rate**: the most common cause of NaN loss. Reduce until training is stable.

4. **Not monitoring gradient norms**: exploding gradients are invisible if you only watch loss. Log gradient norms during training.

5. **Over-relying on default hyperparameters**: initial learning rate, batch size, and architecture interact. What works for image classification may fail for tabular data.

---

## 10. Check Your Understanding

1. Why does a network without activation functions collapse to a linear model?
2. Backpropagation computes $\frac{\partial L}{\partial w}$ for every weight. Why is it more efficient than computing each derivative independently?
3. You use sigmoid as the hidden activation in a 20-layer network. What problem do you expect?
4. Why is cross-entropy the default loss for classification instead of MSE?
5. A network with $1$ hidden layer of width $10$ and ReLU activation has how many parameters if input is 100 and output is 5?

---

## 11. Resumen

Neural networks are compositions of differentiable functions. Each layer transforms its input linearly, then applies a non-linear activation. Training uses backpropagation (chain rule applied efficiently) to compute gradients, then gradient descent to update weights. The key challenges are vanishing/exploding gradients, choosing the right architecture, and preventing overfitting with regularization.

---

## 12. Where to Go Next

- [[Training Techniques]] — Optimizers, schedulers, and practical tricks
- [[CNNs]] — Neural networks for images
- [[Transformers]] — Neural networks for sequences
- [[Calculus]] — The math behind backpropagation
