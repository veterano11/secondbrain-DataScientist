---
tags: [deep-learning, neural-networks, advanced, redes-neuronales]
status: growing
created: 2026-06-27
---

# Redes Neuronales

## 1. Escenario de aprendizaje

Quieres construir un sistema que reconozca dígitos escritos a mano (como los códigos postales en sobres). Tienes imágenes de 28×28 píxeles y quieres que la máquina aprenda a clasificarlas en dígitos del 0 al 9. Las redes neuronales son el motor detrás de esta tarea — desde reconocimiento de imágenes hasta modelos de lenguaje. La idea fundamental es simple: una red neuronal es una **composición de funciones diferenciables** que puede aproximar cualquier función continua (Teorema de Aproximación Universal). Pero la profundidad de la teoría está en entender cómo se componen esas funciones, cómo fluyen los gradientes a través de ellas y cómo la arquitectura afecta lo que la red puede aprender.

---

## 2. El Perceptrón — El Bloque Fundamental

### 2.1 La Neurona

Una sola neurona calcula:

$$y = \sigma(w^T x + b) = \sigma\left(\sum_{i=1}^d w_i x_i + b\right)$$

```
    x₁ ──●── w₁ ──┐
    x₂ ──●── w₂ ──┤
    x₃ ──●── w₃ ──┼──→ Σ ──→ σ ──→ y
    ...           │
    x_d ──●── w_d ─┘
                  ↑
                  b (bias)
```

**Paso a paso**:
1. Cada entrada $x_i$ se multiplica por su peso $w_i$
2. Todas las entradas ponderadas se suman con el sesgo $b$
3. La suma se pasa por una función de activación $\sigma$

**Analogía**: una neurona es una unidad de toma de decisiones. Cada entrada es evidencia a favor o en contra de alguna conclusión ($w_i$ te dice qué tan importante es esa evidencia). El sesgo es la tendencia base. La función de activación convierte la evidencia total en una decisión.

### 2.2 ¿Por qué la No Linealidad?

```
    Sin activación:                Con activación:
    
    x ──→ [Lineal] ──→ y         x ──→ [Lineal] ──→ [σ] ──→ y
         (una capa)                    (red profunda)
    
    Solo puede aprender             Puede aprender
    relaciones lineales             cualquier cosa
```

Si apilamos capas lineales sin funciones de activación, toda la red se reduce a una única transformación lineal:

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W'x + b'$$

Una red lineal multicapa no es más poderosa que una capa simple. **Las funciones de activación introducen no linealidad**, permitiendo a la red aprender relaciones complejas y no lineales.

---

## 3. Funciones de Activación

```
    Sigmoid          Tanh            ReLU           GELU
    y                y               y              y
  1 ┤    ___       1 ┤   __        _│_             _│_
    │  /             │  /          / │             / │
0.5 ┤/              0 ┤/          /  │            /  │
    │                │           /   │           /   │
  0 ┤──────────    -1 ┤─────────    ─┤──────      ─┤──────
    └──────── x      └──────── x     └──── x      └──── x
```

### 3.1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

- **Importancia histórica**: la primera activación ampliamente usada
- **Problema 1 — gradiente que desaparece**: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ tiene un máximo de 0.25. En redes profundas, multiplicar muchos gradientes pequeños causa que la señal desaparezca.
- **Problema 2 — no centrada en cero**: las salidas siempre son positivas, causando gradientes en zigzag
- **Todavía se usa para**: capa de salida de clasificación binaria (salida como probabilidad)

### 3.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$$

- Centrada en cero (arregla el problema de zigzag de sigmoid)
- Todavía sufre de gradiente que desaparece (saturación en ±1)

### 3.3 ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- **La activación por defecto** para capas ocultas
- **Por qué funciona**: la derivada es 1 para $x > 0$ — ¡sin gradiente que desaparezca!
- **Problema**: "ReLU muerto" — si $x < 0$, el gradiente es 0 y la neurona nunca se activa
- **Soluciones**: Leaky ReLU ($\alpha x$ para $x < 0$), PReLU ($\alpha$ aprendible)

### 3.4 GELU (Unidad Lineal de Error Gaussiano)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

donde $\Phi$ es la función de distribución acumulada de la distribución normal estándar.

- Más suave que ReLU, usado en transformers modernos (GPT, BERT, Llama)
- Aproximado como: $\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$

### 3.5 Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}$$

Convierte un vector de $K$ números reales en una distribución de probabilidad (todos positivos, suman 1). Se usa en la **capa de salida para clasificación multiclase**.

---

## 4. Propagación hacia Adelante (Forward Propagation)

Los datos fluyen a través de la red capa por capa:

```
    Entrada          Capa 1          Capa 2         Salida
    (x)              (h₁)            (h₂)           (ŷ)
     │                │               │               │
     ▼                ▼               ▼               ▼
   [x] ──→ [W₁, b₁] ──→ [σ] ──→ [W₂, b₂] ──→ [σ] ──→ [ŷ]
```

$$h_0 = x \quad \text{(entrada)}$$
$$z_l = W_l h_{l-1} + b_l \quad \text{(transformación lineal)}$$
$$h_l = \sigma(z_l) \quad \text{(activación)}$$
$$\hat{y} = h_L \quad \text{(salida)}$$

Cada capa es una transformación lineal seguida de una activación no lineal. La composición de muchas capas permite a la red aprender representaciones jerárquicas — características simples en capas tempranas, características complejas en capas tardías.

---

## 5. Retropropagación (Backpropagation)

### 5.1 El Problema

Tenemos una red con millones de parámetros. Necesitamos el gradiente de la pérdida respecto a **cada** parámetro. Calcular cada derivada independientemente es imposible.

### 5.2 La Solución — Regla de la Cadena

```
    Red neuronal:    x ──→ [W₁] ──→ [σ] ──→ [W₂] ──→ ŷ ──→ L
                                                    │
    Retropropagación:                      dL/dŷ ←──┘
                                              │
                                    dL/dW₂ ←──┤
                                              │
                                    dL/dh₁ ←──┘
                                              │
                                    dL/dW₁ ←──┘
```

La retropropagación aplica la regla de la cadena eficientemente:

1. **Paso hacia adelante**: calcular y almacenar todas las activaciones
2. **Paso hacia atrás**: calcular gradientes desde la última capa hacia atrás, reutilizando valores previamente calculados

**Ejemplo concreto** para una red de 2 capas:

$$L = \frac{1}{2}(y - \hat{y})^2, \quad \hat{y} = W_2 \sigma(W_1 x + b_1) + b_2$$

1. Calcular $\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$
2. Calcular $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2} = (\hat{y} - y) \cdot h_1^T$
3. Calcular $\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial \hat{y}} \cdot W_2^T$
4. Calcular $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial (W_1 x)} \cdot x^T = \frac{\partial L}{\partial h_1} \odot \sigma'(z_1) \cdot x^T$

La idea clave: el gradiente en la capa $l$ depende del gradiente en la capa $l+1$. Al calcular hacia atrás, se reutiliza el gradiente de cada capa.

### 5.3 Diferenciación Automática

Framework como PyTorch y TensorFlow implementan la retropropagación automáticamente (ver [[Optimización Basada en Gradiente]] para más detalles del cálculo de gradientes):

```python
import torch

x = torch.randn(10, requires_grad=True)
y = (x ** 2).sum()
y.backward()                    # calcula todos los gradientes
print(x.grad)                   # d(y)/dx = 2x
```

### 5.4 Gradientes que Desaparecen y Explotan

**Desaparecimiento**: en redes profundas con sigmoid/tanh, los gradientes se vuelven exponencialmente más pequeños a medida que retrocedes. Las capas tempranas aprenden extremadamente lento o nada.

- Solución: ReLU, conexiones residuales, normalización por lotes, inicialización adecuada

**Explosión**: los gradientes se vuelven exponencialmente más grandes en las capas tempranas, causando inestabilidad (pérdida NaN).

- Solución: recorte de gradientes, mejor inicialización, tasa de aprendizaje más baja

---

## 6. Funciones de Pérdida

| Tarea | Función de Pérdida | Fórmula | ¿Por qué esta? |
|---|---|---|---|
| **Regresión** | MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Diferenciable, convexa, penaliza errores grandes |
| **Regresión** | MAE | $\frac{1}{n}\sum|y - \hat{y}|$ | Robusta a valores atípicos |
| **Clasif. binaria** | BCE | $-\frac{1}{n}\sum y\log(\hat{y}) + (1-y)\log(1-\hat{y})$ | Derivada de MLE para Bernoulli |
| **Multiclase** | Entropía cruzada | $-\frac{1}{n}\sum\sum y_c\log(\hat{y}_c)$ | Derivada de MLE para Multinomial |

**¿Por qué entropía cruzada para clasificación?** Es equivalente a maximizar la verosimilitud de las predicciones bajo la distribución real (un concepto central de [[Probabilidad]] / MLE). Minimizar la entropía cruzada = maximizar la probabilidad de la clase correcta.

---

## 7. Inicialización de Pesos

Una inicialización adecuada previene gradientes que desaparecen o explotan al inicio del entrenamiento:

| Inicialización | Distribución | Mejor para |
|---|---|---|
| **Xavier/Glorot** | $\text{Var}(w) = \frac{1}{\text{fan\_in}}$ | Tanh, Sigmoid |
| **He/Kaiming** | $\text{Var}(w) = \frac{2}{\text{fan\_in}}$ | ReLU |
| **Ortogonal** | Matriz ortogonal aleatoria | RNNs |

**Por qué importa**: si los pesos son demasiado grandes, las activaciones explotan. Si son demasiado pequeñas, las activaciones desaparecen. Una inicialización correcta mantiene las activaciones en un rango razonable para todas las capas.

En framework modernos, las inicializaciones por defecto suelen ser suficientes — pero si el entrenamiento es inestable, vale la pena verificar la inicialización.

---

## 8. Principios de Diseño de Arquitectura

```
    Red Pocas Capas          Red Muchas Capas
    (poco profunda)          (muy profunda)
    
    [Capa] → [Capa] → ŷ     [Capa] → [Capa] → ... → [Capa] → ŷ
       │                       │                │
       │                    Skip             Skip
       │                   Connection       Connection
```

- **Ancho vs profundidad**: las redes más profundas aprenden características más abstractas, pero son más difíciles de entrenar. Las redes más anchas aprenden más patrones por capa pero usan más parámetros.
- **Conexiones de salto (skip connections)**: permiten que los gradientes fluyan directamente a las capas tempranas (ResNet). Esenciales para redes con 50+ capas.
- **Normalización**: batch norm, layer norm o group norm estabilizan el entrenamiento.
- **[[Regularización]]**: dropout, peso de decaimiento, parada temprana previenen el sobreajuste.

---

## 9. Errores Comunes

1. **ReLU en la capa de salida**: ReLU produce $[0, \infty)$, lo cual no tiene sentido como probabilidad o valor de regresión sin restricción. Usar lineal (regresión), sigmoid (binaria) o softmax (multiclase).

2. **Sin normalización**: las redes profundas sin normalización son inestables. Usar batch norm o layer norm.

3. **Tasa de aprendizaje demasiado alta**: la causa más común de pérdida NaN. Reducir hasta que el entrenamiento sea estable.

4. **No monitorear normas de gradiente**: los gradientes que explotan son invisibles si solo observas la pérdida. Registrar normas de gradiente durante el entrenamiento.

5. **Dependencia excesiva de hiperparámetros por defecto**: la tasa de aprendizaje inicial, el tamaño de lote y la arquitectura interactúan. Lo que funciona para clasificación de imágenes puede fallar para datos tabulares.

---

## 10. Verifica tu Comprensión

1. ¿Por qué una red sin funciones de activación colapsa a un modelo lineal?
2. La retropropagación calcula $\frac{\partial L}{\partial w}$ para cada peso. ¿Por qué es más eficiente que calcular cada derivada independientemente?
3. Usas sigmoid como activación oculta en una red de 20 capas. ¿Qué problema esperas?
4. ¿Por qué la entropía cruzada es la pérdida por defecto para clasificación en lugar de MSE?
5. Una red con $1$ capa oculta de ancho $10$ y activación ReLU tiene ¿cuántos parámetros si la entrada es 100 y la salida es 5?

---

## 11. Resumen

Las redes neuronales son composiciones de funciones diferenciables. Cada capa transforma su entrada linealmente, luego aplica una activación no lineal. El entrenamiento usa retropropagación (regla de la cadena aplicada eficientemente) para calcular gradientes, luego descenso por gradiente para actualizar pesos. Los desafíos clave son los gradientes que desaparecen o explotan, elegir la arquitectura correcta, y prevenir el sobreajuste con regularización.

---

## 12. Dónde Ir Ahora

- [[Técnicas de Entrenamiento]] — Optimizadores, planificadores y trucos prácticos
- [[CNNs]] — Redes neuronales para imágenes
- [[Transformers]] — Redes neuronales para secuencias
- [[Cálculo]] — Las matemáticas detrás de la retropropagación
