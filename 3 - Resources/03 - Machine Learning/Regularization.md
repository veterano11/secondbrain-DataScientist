---
tags: [machine-learning, regularization, core]
status: growing
created: 2026-06-27
---

# Regularization

## 1. Escenario de aprendizaje

Estás entrenando un modelo para predecir el precio de casas usando 300 características — metros cuadrados, número de habitaciones, año de construcción, distancia al centro, etc. Tu modelo logra un error casi nulo en entrenamiento, pero cuando lo pruebas con casas nuevas, las predicciones son pésimas. El modelo ha memorizado el ruido y los detalles irrelevantes de los datos de entrenamiento en lugar de aprender patrones generales. Regularization es el conjunto de técnicas para **prevenir este overfitting** restringiendo la complejidad del modelo.

La idea central: un modelo más simple es mejor que uno complejo, todo lo demás siendo igual (navaja de Occam). La regularización penaliza la complejidad, empujando al modelo hacia soluciones más simples que generalizan mejor.

---

## 2. The Intuition — Why Constrain the Model?

Imagine fitting a polynomial to 5 points:
- Degree 1 (line): underfits (high bias, low variance)
- Degree 4 (perfect fit): overfits (low bias, high variance)
- Degree 2-3: balanced

Regularization "penalizes" large coefficients. A model with smaller coefficients is simpler — it does not change as dramatically with small input changes, making it more stable and less likely to overfit.

---

## 3. L2 Regularization (Ridge)

### 3.1 The Math

Add the sum of squared weights to the loss function:

$$L_{\text{ridge}}(w) = \underbrace{\|y - Xw\|^2}_{\text{original loss}} + \underbrace{\lambda \|w\|_2^2}_{\text{penalty}}$$

The gradient update becomes:

$$w_{t+1} = w_t - \eta (\nabla L_{\text{original}} + 2\lambda w_t)$$

Each step shrinks weights by $2\lambda \eta w_t$ — **weight decay**. See [[Gradient-Based Optimization]] for more on gradient descent variants.

### 3.2 Intuition

- Large weights are penalized more (squared penalty)
- Weights shrink toward zero but **never reach exactly zero**
- All features remain in the model, just with smaller coefficients
- Especially useful when features are correlated (it keeps all of them but distributes coefficients evenly)

### 3.3 Effect on Different Models

| Model | Ridge Effect |
|---|---|
| **Linear regression** | Shrinks coefficients, reduces variance |
| **Logistic regression** | Smoother decision boundary |
| **Neural networks** | Weight decay (standard name for L2 in DL) |

### 3.4 Choosing $\lambda$

- $\lambda = 0$: no regularization (original model)
- $\lambda \to \infty$: weights → 0 (only intercept remains)
- Pick $\lambda$ via cross-validation:

```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
model.fit(X, y)
print(model.alpha_)  # best lambda
```

---

## 4. L1 Regularization (Lasso)

### 4.1 The Math

Add the sum of absolute weights:

$$L_{\text{lasso}}(w) = \|y - Xw\|^2 + \lambda \|w\|_1$$

### 4.2 Why Lasso Produces Sparse Solutions

This is the key insight. The L1 penalty has a **sharp corner at zero** where the derivative is discontinuous. During optimization, weights hit exactly zero and stay there.

**Geometric intuition**: the L1 penalty is a diamond-shaped constraint region in weight space. The optimal solution often lies at a corner of this diamond, where some weights are exactly zero.

**Practical consequence**: Lasso does **automatic feature selection** — it drives irrelevant features to exactly zero, leaving only the important ones. See [[Feature Engineering]] for more selection methods.

### 4.3 L1 vs L2

| Property | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalty | $\sum |w_i|$ | $\sum w_i^2$ |
| Effect | Sparse (many zeros) | Shrinkage (all small) |
| Feature selection | Yes (drops features) | No (keeps all features) |
| Correlated features | Picks one arbitrarily | Keeps all, shrinks evenly |
| Sensitivity to outliers | More robust | Less robust |
| Gradient | Constant (±1) | Proportional to $w$ |

### 4.4 Elastic Net

Best of both worlds:

$$L_{\text{elastic}}(w) = \|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

When features are correlated, Lasso picks one at random. Elastic Net tends to select groups of correlated features together — which is often what you want.

---

## 5. Dropout (Neural Networks)

### 5.1 How It Works

During each training iteration, randomly "drop" (set to zero) a fraction $p$ of neurons:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),    # 50% chance of dropping each neuron
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 10),
)
```

### 5.2 Why Dropout Works

Dropout forces the network to learn **redundant representations** — no single neuron can be essential because it might be dropped at any time. The result: the network learns robust features that work with and without any particular neuron.

At test time, dropout is turned off, and all neurons contribute. The weights are scaled by $1-p$ to compensate.

**Analogy**: it is like training an ensemble of $2^n$ sub-networks (where $n$ is the number of neurons) and averaging them at test time.

---

## 6. Early Stopping

### 6.1 How It Works

Monitor validation loss during training. Stop when it stops improving. See [[Training Techniques]] for more on training loop best practices.

```python
# PyTorch-like pseudocode
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model)           # save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:  # e.g., patience=5
            break
```

### 6.2 Why Early Stopping Works

As training progresses, the model first learns general patterns (reducing both train and val loss), then starts overfitting to noise (train loss continues decreasing, val loss increases). Early stopping catches the model right before it starts overfitting.

---

## 7. Data Augmentation

Generate synthetic training data by creating realistic variations:

**For images**: rotation, flip, crop, color jitter, noise, cutout
```python
from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomResizedCrop(224),
])
```

**For text**: back-translation (translate to another language and back), word dropout, synonym replacement, character noise

**Why it works**: each augmented sample is slightly different from the original, teaching the model to be invariant to irrelevant variations.

---

## 8. Batch Normalization

Normalizes activations across each mini-batch:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- **Reduces internal covariate shift**: each layer sees normalized inputs
- **Allows higher learning rates**: gradients are better behaved
- **Acts as a regularizer**: the noise from batch statistics adds slight regularization
- **Makes deep networks trainable**: enables 50+ layer networks

---

## 9. Common Mistakes

1. **Applying L1/L2 to tree-based models**: trees do not have "coefficients" in the same sense. Use max_depth, min_samples_leaf, and other tree-specific parameters instead.

2. **Cross-validating $\lambda$ on the same data used for evaluation**: this leaks information. Always cross-validate on training data only, then evaluate on held-out test data.

3. **Setting $\lambda$ too high**: a model with all weights near zero predicts the mean/constant — trivial and useless.

4. **Dropout after every layer**: too much dropout prevents learning entirely. Use higher dropout on larger layers, lower (or none) on small layers.

5. **Forgetting to scale dropout at test time**: during inference, dropout must be turned off and weights scaled (PyTorch/TF handle this automatically if you call `model.eval()`).

---

## 10. Check Your Understanding

1. You run Lasso with $\lambda = 0.1$ and 50 features become zero. What happens if you increase $\lambda$ to 1.0? What if you decrease to 0.01?

2. Ridge shrinks all weights proportionally. Lasso drives some to zero. Why does L1 produce sparsity but L2 does not? (Hint: think about the shape of the constraint region.)

3. Dropout rate $p=0.5$ means each neuron has a 50% chance of being zeroed. What happens at test time? How are the weights adjusted?

4. Early stopping requires a validation set. What happens if you use the test set for early stopping? Why?

5. You train a neural network and observe that training loss decreases but validation loss increases after epoch 10. What do you do?

---

## 11. Summary

Regularization prevents overfitting by constraining the model. L1 (Lasso) drives weights to zero (feature selection). L2 (Ridge) shrinks weights uniformly (stability). Elastic Net combines both. Dropout and early stopping are specialized for neural networks. Data augmentation and batch normalization also provide regularization effects. The key insight: **a simpler model generalizes better**. Regularization is how you enforce simplicity.

---

## 12. Where to Go Next

- [[Supervised Learning]] — Applying regularization to regression/classification
- [[Model Evaluation]] — Detecting overfitting that regularization should fix
- [[Neural Networks]] — Dropout, weight decay, and batch norm in DL
