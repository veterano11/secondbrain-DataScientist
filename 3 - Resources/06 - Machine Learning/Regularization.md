---
tags: [machine-learning, regularization, core]
status: growing
created: 2026-06-27
---

# Regularización

## 1. Escenario de aprendizaje

Estás entrenando un modelo para predecir el precio de casas usando 300 características — metros cuadrados, número de habitaciones, año de construcción, distancia al centro, etc. Tu modelo logra un error casi nulo en entrenamiento, pero cuando lo pruebas con casas nuevas, las predicciones son pésimas. El modelo ha memorizado el ruido y los detalles irrelevantes de los datos de entrenamiento en lugar de aprender patrones generales. La regularización es el conjunto de técnicas para **prevenir este overfitting** restringiendo la complejidad del modelo.

La idea central: un modelo más simple es mejor que uno complejo, todo lo demás siendo igual (navaja de Occam). La regularización penaliza la complejidad, empujando al modelo hacia soluciones más simples que generalizan mejor.

---

## 2. La Intuición — ¿Por qué Restringir el Modelo?

Imagina ajustar un polinomio a 5 puntos:
- Grado 1 (línea): underfitting (alto sesgo, baja varianza)
- Grado 4 (ajuste perfecto): overfitting (bajo sesgo, alta varianza)
- Grado 2-3: equilibrado

La regularización "penaliza" coeficientes grandes. Un modelo con coeficientes más pequeños es más simple — no cambia dramáticamente con pequeñas variaciones de entrada, haciéndolo más estable y menos propenso a overfitting.

---

## 3. Regularización L2 (Ridge)

### 3.1 La Matemática

Añadir la suma de pesos cuadrados a la función de pérdida:

$$L_{\text{ridge}}(w) = \underbrace{\|y - Xw\|^2}_{\text{pérdida original}} + \underbrace{\lambda \|w\|_2^2}_{\text{penalización}}$$

La actualización del gradiente se convierte:

$$w_{t+1} = w_t - \eta (\nabla L_{\text{original}} + 2\lambda w_t)$$

Cada paso reduce los pesos en $2\lambda \eta w_t$ — **decaimiento de pesos**. Ver [[Gradient-Based Optimization]] para más sobre variantes de descenso de gradiente.

### 3.2 Intuición

- Los pesos grandes se penalizan más (penalización cuadrada)
- Los pesos se reducen hacia cero pero **nunca llegan exactamente a cero**
- Todas las features permanecen en el modelo, solo con coeficientes más pequeños
- Especialmente útil cuando las features están correlacionadas (mantiene todas pero distribuye coeficientes uniformemente)

### 3.3 Efecto en Diferentes Modelos

| Modelo | Efecto Ridge |
|---|---|
| **Regresión lineal** | Reduce coeficientes, disminuye varianza |
| **Regresión logística** | Frontera de decisión más suave |
| **Redes neuronales** | Decaimiento de pesos (nombre estándar para L2 en DL) |

### 3.4 Elección de $\lambda$

- $\lambda = 0$: sin regularización (modelo original)
- $\lambda \to \infty$: pesos → 0 (solo queda la intersección)
- Elegir $\lambda$ vía validación cruzada:

```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
model.fit(X, y)
print(model.alpha_)  # mejor lambda
```

---

## 4. Regularización L1 (Lasso)

### 4.1 La Matemática

Añadir la suma de pesos absolutos:

$$L_{\\text{lasso}}(w) = \|y - Xw\|^2 + \lambda \|w\|_1$$

### 4.2 Por qué Lasso Produce Soluciones Dispersas

Esta es la idea clave. La penalización L1 tiene una **esquina pronunciada en cero** donde la derivada es discontinua. Durante la optimización, los pesos llegan exactamente a cero y se quedan allí.

**Intuición geométrica**: la penalización L1 es una región de restricción con forma de diamante en el espacio de pesos. La solución óptima a menudo yace en una esquina de este diamante, donde algunos pesos son exactamente cero.

**Consecuencia práctica**: Lasso hace **selección automática de features** — impulsa features irrelevantes a exactamente cero, dejando solo las importantes. Ver [[Feature Engineering]] para más métodos de selección.

### 4.3 L1 vs L2

| Propiedad | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalización | $\sum |w_i|$ | $\sum w_i^2$ |
| Efecto | Disperso (muchos ceros) | Reducción (todos pequeños) |
| Selección de features | Sí (elimina features) | No (mantiene todas) |
| Features correlacionadas | Elige una arbitrariamente | Mantiene todas, reduce uniformemente |
| Sensibilidad a outliers | Más robusto | Menos robusto |
| Gradiente | Constante (±1) | Proporcional a $w$ |

### 4.4 Elastic Net

Lo mejor de ambos mundos:

$$L_{\text{elastic}}(w) = \|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

Cuando las features están correlacionadas, Lasso elige una al azar. Elastic Net tiende a seleccionar grupos de features correlacionadas juntas — lo cual a menudo es lo que quieres.

---

## 5. Dropout (Redes Neuronales)

### 5.1 Cómo Funciona

Durante cada iteración de entrenamiento, "eliminar" aleatoriamente (poner en cero) una fracción $p$ de neuronas:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),    # 50% de probabilidad de eliminar cada neurona
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 10),
)
```

### 5.2 Por qué Dropout Funciona

Dropout fuerza a la red a aprender **representaciones redundantes** — ninguna neurona individual puede ser esencial porque podría ser eliminada en cualquier momento. El resultado: la red aprende features robustas que funcionan con y sin cualquier neurona particular.

En tiempo de prueba, dropout se desactiva y todas las neuronas contribuyen. Los pesos se escalan por $1-p$ para compensar.

**Analogía**: es como entrenar un ensemble de $2^n$ sub-redes (donde $n$ es el número de neuronas) y promediarlas en tiempo de prueba.

---

## 6. Parada Temprana

### 6.1 Cómo Funciona

Monitorear la pérdida de validación durante el entrenamiento. Parar cuando deja de mejorar. Ver [[Training Techniques]] para más sobre mejores prácticas del bucle de entrenamiento.

```python
# Pseudocódigo estilo PyTorch
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model)           # guardar mejor modelo
    else:
        patience_counter += 1
        if patience_counter >= patience:  # ej., patience=5
            break
```

### 6.2 Por qué la Parada Temprana Funciona

A medida que avanza el entrenamiento, el modelo primero aprende patrones generales (reduciendo tanto pérdida de entrenamiento como de validación), luego comienza a hacer overfitting al ruido (la pérdida de entrenamiento sigue disminuyendo, la de validación aumenta). La parada temprana atrapa al modelo justo antes de que empiece a hacer overfitting.

---

## 7. Aumentación de Datos

Generar datos de entrenamiento sintéticos creando variaciones realistas:

**Para imágenes**: rotación, volteo, recorte, jitter de color, ruido, cutout
```python
from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomResizedCrop(224),
])
```

**Para texto**: retrotraducción (traducir a otro idioma y volver), eliminación de palabras, reemplazo de sinónimos, ruido de caracteres

**Por qué funciona**: cada muestra aumentada es ligeramente diferente de la original, enseñando al modelo a ser invariante a variaciones irrelevantes.

---

## 8. Normalización por Lotes

Normaliza activaciones a través de cada mini-lote:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- **Reduce el desplazamiento de covariable interno**: cada capa ve entradas normalizadas
- **Permite tasas de aprendizaje más altas**: los gradientes se comportan mejor
- **Actúa como regularizador**: el ruido de las estadísticas del lote añade ligera regularización
- **Hace entrenables las redes profundas**: permite redes de 50+ capas

---

## 9. Errores Comunes

1. **Aplicar L1/L2 a modelos basados en árboles**: los árboles no tienen "coeficientes" en el mismo sentido. Usar max_depth, min_samples_leaf y otros parámetros específicos de árboles.

2. **Hacer validación cruzada de $\lambda$ en los mismos datos usados para evaluación**: esto filtra información. Siempre hacer validación cruzada solo en datos de entrenamiento, luego evaluar en datos de prueba apartados.

3. **Poner $\lambda$ demasiado alto**: un modelo con todos los pesos cerca de cero predice la media/constante — trivial e inútil.

4. **Dropout después de cada capa**: demasiado dropout impide el aprendizaje por completo. Usar dropout mayor en capas más grandes, menor (o ninguno) en capas pequeñas.

5. **Olvidar escalar dropout en tiempo de prueba**: durante la inferencia, dropout debe desactivarse y los pesos escalarse (PyTorch/TF manejan esto automáticamente si llamas `model.eval()`).

---

## 10. Comprueba tu Conocimiento

1. Ejecutas Lasso con $\lambda = 0.1$ y 50 features se vuelven cero. ¿Qué pasa si aumentas $\lambda$ a 1.0? ¿Y si disminuyes a 0.01?

2. Ridge reduce todos los pesos proporcionalmente. Lasso impulsa algunos a cero. ¿Por qué L1 produce dispersión pero L2 no? (Pista: piensa en la forma de la región de restricción.)

3. Tasa de dropout $p=0.5$ significa que cada neurona tiene 50% de probabilidad de ser puesta en cero. ¿Qué pasa en tiempo de prueba? ¿Cómo se ajustan los pesos?

4. La parada temprana requiere un conjunto de validación. ¿Qué pasa si usas el conjunto de prueba para la parada temprana? ¿Por qué?

5. Entrenas una red neuronal y observas que la pérdida de entrenamiento disminuye pero la de validación aumenta después de la época 10. ¿Qué haces?

---

## 11. Resumen

La regularización previene el overfitting restringiendo el modelo. L1 (Lasso) impulsa pesos a cero (selección de features). L2 (Ridge) reduce pesos uniformemente (estabilidad). Elastic Net combina ambos. Dropout y parada temprana están especializados para redes neuronales. La aumentación de datos y la normalización por lotes también proporcionan efectos de regularización. La idea clave: **un modelo más simple generaliza mejor**. La regularización es cómo enforceas la simplicidad.

---

## 12. ¿Dónde ir Siguente?

- [[Supervised Learning]] — Aplicar regularización a regresión/clasificación
- [[Model Evaluation]] — Detectar overfitting que la regularización debería corregir
- [[Neural Networks]] — Dropout, decaimiento de pesos y normalización por lotes en DL
