---
tags: [optimization, gradient-descent, sgd, deep-learning]
status: growing
created: 2026-06-27
---

# Gradient-Based Optimization

## 1. Escenario de aprendizaje

Estás entrenando una red neuronal. La loss baja rápido al principio, luego se estanca. Cambiás el learning rate y ahora la loss diverge a NaN. Probás Adam y funciona mejor que SGD, pero no sabés por qué.

Gradient-based optimization es el motor del deep learning. Este note te da el criterio para elegir optimizador, learning rate, y scheduler.

## 2. Gradient Descent

La idea más simple: calcular la pendiente de la loss en el punto actual y moverse en dirección opuesta.

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

El gradiente $\nabla \mathcal{L}$ es la derivada del [[Calculus]] de la loss respecto a los parámetros.

```python
for epoch in range(100):
    grad = compute_gradient(loss, params, data)  # todo el dataset
    params -= lr * grad
```

**Problema**: si el dataset tiene 1M de muestras, cada paso procesa 1M ejemplos. Muy lento.

## 3. SGD (Stochastic Gradient Descent)

Usa una muestra a la vez o mini-batches:

```python
for X_batch, y_batch in dataloader:  # batch de 32-256
    loss = model(X_batch, y_batch)
    loss.backward()
    optimizer.step()  # params -= lr * grad
```

**El ruido de SGD es beneficioso**: ayuda a escapar de saddle points y mínimos locales chicos. SGD es el motor del [[Supervised Learning]] moderno.

## 4. Momentum

Acumula velocidad: si los gradientes apuntan consistentemente en la misma dirección, acelera. Si cambian de dirección, frena.

$$v_{t+1} = \mu v_t + \eta \nabla \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Analogía**: una pelota bajando una colina. Gana velocidad en pendientes constantes y frena en valles. Los gradientes son vectores; las operaciones de actualización son de [[Linear Algebra]].

## 5. Adam

Combina momentum + adaptación por parámetro. Es el optimizador default para la mayoría de problemas.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

Adam ajusta el learning rate por parámetro: parámetros con gradientes grandes → learning rate chico. Parámetros con gradientes chicos → learning rate grande.

**Hiperparámetros por defecto** (funcionan casi siempre):

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| lr | 3e-4 | Learning rate |
| $\beta_1$ | 0.9 | Momento del gradiente |
| $\beta_2$ | 0.999 | Momento del gradiente al cuadrado |
| $\epsilon$ | 1e-8 | Estabilidad numérica |

### ¿Adam vs SGD?

```text
Problema: clasificación CIFAR-10 con ResNet-18
Adam:     converge más rápido (50 epochs para 90%)
SGD+momentum: converge más lento (100 epochs) pero llega a 92%

Regla práctica:
- Prototipado rápido → Adam
- Producción, querés el mejor score → SGD + momentum + learning rate schedule
```

## 6. Learning Rate Schedules

El LR óptimo no es constante. Al principio, querés pasos grandes. Al final, pasos chicos para no saltarte el mínimo.

```python
# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Cada 30 epochs: lr *= 0.1

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# lr = lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / T_max)) / 2

# Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
# Si loss no mejora en 5 epochs → lr *= 0.5

# Linear warmup (para transformers)
def warmup_scheduler(epoch, warmup=5, lr_max=3e-4):
    if epoch < warmup:
        return (epoch + 1) / warmup  # lr crece linealmente
    return 1.0
```

**LR Range Test**: encontrá el mejor LR corriendo un epoch con LR creciente:

```python
# Aumentá lr linealmente de 1e-7 a 10, registrá la loss
# El mejor LR está en el punto de mayor pendiente descendente
```

## 7. Gradient Clipping

Para RNNs y Transformers, los gradientes pueden explotar:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Escala los gradientes si su norma > 1.0
```

## 8. AdamW

Corrige Adam: la regularización L2 (weight decay) en Adam se aplica mal. AdamW separa weight decay del update adaptativo.

```python
# AdamW (PyTorch)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

## 9. Common Mistakes

1. **Learning rate muy alto**: loss → NaN. Bajá LR 10×.
2. **Learning rate muy bajo**: loss baja muy lentamente. Hacé LR range test.
3. **No schedule**: SGD sin schedule nunca converge óptimamente. Probar cosine annealing.
4. **Adam sin warmup en transformers**: los primeros pasos de Adam pueden ser muy grandes con embeddings recién inicializados. Warmup 5-10% del training.
5. **No monitor ear gradient norm**: si la norma del gradiente crece durante el entrenamiento, eventualmente llega a NaN. Clippeá.

## 10. Check Your Understanding

1. ¿Por qué SGD con momentum converge mejor que Adam en algunos problemas? (SGD encuentra soluciones más llanas que generalizan mejor)
2. ¿Cuándo usarías gradient clipping? (RNNs, Transformers, problemas donde los gradientes explotan)
3. ¿Qué hace cosine annealing? (Decae el LR siguiendo una función coseno, dando múltiples "reinicios" suaves)

## 11. Resumen

Gradient-based optimization mueve parámetros en la dirección opuesta al gradiente. SGD con momentum es el clásico que mejor generaliza. Adam es el default rápido y robusto. Los LR schedules (step, cosine, warmup) son esenciales para buen rendimiento. Gradient clipping evita la divergencia. AdamW es la versión moderna de Adam con weight decay correcto. La técnica más importante: encontrar el LR correcto con LR range test.

## 12. Where to Go Next

- [[Convex Optimization]] — teoría de convergencia
- [[Hyperparameter Tuning]] — optimizar LR, batch size, weight decay
- [[Training Techniques]] — batch norm, dropout, init
- [[Neural Networks]] — backpropagation, el engine del gradiente
