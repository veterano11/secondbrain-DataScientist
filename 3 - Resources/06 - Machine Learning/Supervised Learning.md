---
tags: [machine-learning, supervised, core, aprendizaje-supervisado]
status: growing
created: 2026-06-27
---

# Aprendizaje Supervisado

## 1. Escenario de aprendizaje

Imagina que trabajas en un banco y te piden predecir si un cliente va a dejar el servicio (churn) con base en su historial de transacciones, llamadas al soporte y productos contratados. Tienes miles de ejemplos de clientes pasados donde ya sabes si se fueron o se quedaron. Eso es supervised learning: aprendes un mapeo de entradas $x$ (datos del cliente) a salidas $y$ (churn o no churn) usando ejemplos etiquetados $(x_i, y_i)$. Pero la simplicidad del concepto esconde la profundidad de los algoritmos, la sutileza de la evaluación y el cuidado necesario para evitar overfitting.

---

## 2. El Marco de Aprendizaje Supervisado

### 2.1 Definición Formal

Dado:
- Datos de entrenamiento $(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)$
- $x_i \in \mathcal{X}$ (espacio de características)
- $y_i \in \mathcal{Y}$ (espacio de etiquetas: $\mathbb{R}$ para regresión, $\{1,...,K\}$ para clasificación)

Encontrar $f: \mathcal{X} \to \mathcal{Y}$ que minimice la pérdida esperada en datos nuevos.

### 2.2 Las Dos Ramas Principales

```
                    Aprendizaje Supervisado
                           │
              ┌────────────┴────────────┐
              │                         │
         Regresión                 Clasificación
     (y continuo)               (y discreto)
              │                         │
    ┌─────────┴─────────┐     ┌─────────┴─────────┐
    │                   │     │                   │
Pronosticar        Temperatura  Detectar         Diagnosticar
precio casa        ventas       spam             enfermedad
```

| Rama | Tipo de $y$ | Ejemplo |
|---|---|---|
| **Regresión** | Continuo | Predecir precio de vivienda, temperatura, ventas |
| **Clasificación** | Discreto | Detectar spam, clasificar imágenes, diagnosticar enfermedad |

---

## 3. Modelos Lineales

### 3.1 Regresión Lineal

El modelo más simple e interpretable. Asume una relación lineal entre las características y el objetivo:

$$y = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b + \epsilon = Xw + \epsilon$$

```
    y
    │        ╱  ← recta de regresión
    │      ╱
    │    ╱ • datos reales
    │  ╱
    │╱
    └──────────── x
```

**Intuición geométrica**: encontrar un hiperplano que mejor se ajuste a los puntos de datos.

**La Ecuación Normal** (solución cerrada — ver [[Álgebra Lineal]] para detalles de inversión de matrices):

$$\hat{w} = (X^T X)^{-1} X^T y$$

**Paso a paso**:
1. $X^T X$: calcular la matriz de covarianza (d × d)
2. $(X^T X)^{-1}$: invertirla (requiere rango completo — sin multicolinealidad)
3. $X^T y$: calcular la correlación entre características y objetivo
4. Multiplicar: $\hat{w}$ da los pesos óptimos

**Ejemplo concreto** — predecir salario según años de experiencia:
```
Datos: (1año, 40k), (2años, 45k), (3años, 50k), (4años, 55k)

X = [[1], [2], [3], [4]]
y = [40000, 45000, 50000, 55000]

w = (X^T X)^(-1) X^T y
  = (suma x_i²)^(-1) × suma(x_i y_i)
  = 1/30 × 150000 = 5000

Entonces: salario ≈ 35000 + 5000 × años_experiencia
```

**Supuestos** (vale la pena conocer porque violarlos degrada el rendimiento — ver [[Estadística]] para contexto más profundo):
1. **Linealidad**: la relación entre características y objetivo es lineal
2. **Independencia**: las observaciones son independientes (sin autocorrelación)
3. **Homocedasticidad**: varianza constante de errores en todos los valores de x
4. **Normalidad**: los errores están distribuidos normalmente (para inferencia, no predicción)

Cuando estos supuestos se violan, usa regularización (Ridge/Lasso), modelos no lineales o transformaciones.

### 3.2 Regresión Logística

A pesar del nombre, es un algoritmo de **clasificación**. Modela la probabilidad de pertenecer a una clase:

$$P(y=1|x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$

**¿Por qué sigmoid?** La regresión lineal produce valores sin restricción $(-\infty, \infty)$. Una probabilidad debe estar en $[0, 1]$. La función sigmoid comprime cualquier número real en este rango:

```
 σ(z)
  1 ┤                 ___________
    │               /
    │             /
0.5 ┤─ ─ ─ ─ ─ ─/─ ─ ─ ─ ─ ─ ─  ← frontera de decisión
    │          /
    │        /
  0 ┤________/
    └──────────────────────────── z
         -∞    0    +∞
```

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

- Cuando $z \to \infty$: $\sigma(z) \to 1$
- Cuando $z \to -\infty$: $\sigma(z) \to 0$
- Cuando $z = 0$: $\sigma(0) = 0.5$ (frontera de decisión)

**Frontera de decisión**: el conjunto de puntos donde $P(y=1|x) = 0.5$, es decir, $w^T x = 0$. Esta es una superficie lineal.

**Extensión multiclase**: regresión softmax (también llamada regresión logística multinomial):

$$P(y=k|x) = \frac{e^{w_k^T x}}{\sum_{j=1}^K e^{w_j^T x}}$$

---

## 4. Modelos Basados en Árboles

### 4.1 Árboles de Decisión

Un árbol de decisión divide los datos recursivamente basándose en valores de características.

```
              ¿Edad > 30?
              /          \
            Sí            No
            /              \
    ¿Ingresos > 50k?   ¿Tiene hogar?
      /       \          /       \
    Sí        No        Sí        No
   [Aprobado] [Rechazado] [Aprobado] [Rechazado]
```

**Cómo funciona** (paso a paso):
1. Observa todas las características y todos los puntos de división posibles
2. Elige la división que mejor separa el objetivo (menor impureza)
3. Repite recursivamente en cada partición
4. Detén cuando se alcanza la profundidad máxima, se cumple el mínimo de muestras por hoja, o ninguna división mejora la pureza

**Métricas de impureza**:
- **Gini**: $2p(1-p)$ para clasificación binaria
- **Entropía**: $-p\log p - (1-p)\log(1-p)$
- **MSE** (regresión): varianza del objetivo en el nodo

**Intuición**: imagina clasificar correos por "contiene la palabra 'gratis'" — la primera división los separa en dos grupos, uno con mayormente spam (izquierda) y otro con mayormente correos legítimos (derecha). Ahora divide el grupo izquierdo por "contiene la palabra 'urgente'", y así sucesivamente.

### 4.2 Bosque Aleatorio (Random Forest)

Construye muchos árboles de decisión y promedia sus predicciones.

**Por qué funciona**: cada árbol se entrena con una muestra bootstrap diferente de los datos (bagging), y cada división considera solo un subconjunto aleatorio de características. Esto descorrelaciona los árboles. El promedio de muchos árboles imperfectos es más estable y preciso que cualquier árbol individual.

```
    Datos de entrenamiento
            │
    ┌───────┼───────┐
    │       │       │
  Muestra  Muestra  Muestra
  Bootstrap Bootstrap Bootstrap
    │       │       │
  Árbol 1  Árbol 2  Árbol 3
    │       │       │
    └───────┼───────┘
            │
       Promedio
            │
      Predicción Final
```

- **Reduce la varianza** sin aumentar significativamente el sesgo
- **Maneja no linealidad** de forma natural
- **Importancia de características**: las características usadas cerca de la parte superior de muchos árboles son más importantes

### 4.3 Boosting por Gradiente (XGBoost, LightGBM)

**Intuición**: en lugar de promediar muchos árboles independientes (Random Forest), construir árboles **secuencialmente**, donde cada árbol intenta corregir los errores de los anteriores.

1. Empezar con una predicción simple (ej. media del objetivo)
2. Calcular residuos (errores) de la predicción actual
3. Entrenar un árbol pequeño para predecir los residuos
4. Agregar la predicción del árbol al ensemble (con una tasa de aprendizaje)
5. Repetir los pasos 2-4 cientos o miles de veces

XGBoost agrega regularización a los árboles, maneja valores faltantes y está altamente optimizado para rendimiento vía [[Optimización Basada en Gradiente]]. Es el algoritmo predilecto para competiciones de datos tabulares.

---

## 5. Máquinas de Soporte Vectorial (SVM)

**Idea central**: encontrar el hiperplano que separa las clases con el **máximo margen**.

```
    Clase A (●)              Clase B (○)
         ●                        ○
       ● ●         MARGEN       ○ ○
     ● ● ● │                 │ ○ ○ ○
       ● ● │   ●  ← soporte │ ○ ○
         ● │   vectorial     ○
           │                 │
    ───────┼─────────────────┼─────── Hiperplano
           │                 │
         ● │   ●  ← soporte │ ○
       ● ● │   vectorial     ○ ○
     ● ● ● │                 │ ○ ○ ○
       ● ●         MARGEN       ○ ○
         ●                        ○
```

El margen es la distancia desde el hiperplano hasta los puntos más cercanos de cada clase (los vectores de soporte). Maximizar el margen mejora la generalización.

**Truco del kernel**: mapear datos a un espacio de mayor dimensionalidad donde se vuelven linealmente separables, sin calcular explícitamente el mapeo:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

Kernels comunes:
- **Lineal**: $K(x_i, x_j) = x_i^T x_j$ — sin mapeo, solo SVM lineal
- **RBF** (Gaussiano): $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ — mapeo de dimensión infinita
- **Polinomial**: $K(x_i, x_j) = (x_i^T x_j + c)^d$

Las SVM funcionan bien cuando el número de características es grande en relación con las muestras (ej. clasificación de texto con bolsa de palabras).

---

## 6. K-Vecinos Más Cercanos (KNN)

El algoritmo más simple: almacenar todos los datos de entrenamiento. Para predecir un nuevo punto, encontrar los $k$ puntos de entrenamiento más cercanos y votar.

```
    ¿A qué clase pertenece ★?
    
    ● ● ●
    ● ★ ●     → K=3: 2● vs 1○ → Clase ●
    ● ● ○
      ○ ○
      ○
```

- **Sin entrenamiento** (aprendiz perezoso) — solo memorizar los datos
- La predicción es O(n) — debe calcular la distancia a cada punto de entrenamiento
- Sensible al escalado de características (usar StandardScaler)
- Funciona mejor con pocas características (maldición de la dimensionalidad)

---

## 7. Evaluando Modelos Supervisados

La división entrenamiento/prueba es innegociable:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # para clasificación
)
```

Para una evaluación confiable:
- Siempre usar un conjunto de prueba aparte (no solo validación)
- Usar validación cruzada para [[Ajuste de Hiperparámetros]]
- Estratificar las divisiones de clasificación para preservar las proporciones de clases

---

## 8. Errores Comunes

1. **Fuga de datos**: usar información del conjunto de entrenamiento durante el entrenamiento. Formas comunes: escalar antes de dividir, usar datos futuros para predecir el pasado, características a nivel de grupo sin división de grupos.

2. **Entrenar en datos desbalanceados sin cuidado**: 99% de precisión en un desbalance de 99:1 no tiene sentido. Usar pesos de clase, remuestreo (SMOTE) o diferentes métricas (F1, precisión-recall AUC).

3. **Usar precisión en problemas desbalanceados**: un modelo que predice "no enfermedad" para todos logra 99% de precisión cuando la prevalencia de la enfermedad es 1%. Siempre verificar la matriz de confusión.

4. **No verificar supuestos del modelo**: la regresión lineal asume linealidad y homocedasticidad. Si se violan, las predicciones pueden estar sistemáticamente sesgadas.

5. **Sobreajuste antes de ver datos de prueba**: ajustar hiperparámetros en el conjunto de prueba lo invalida. Usar un conjunto de validación separado o validación cruzada.

---

## 9. Verifica tu Comprensión

1. ¿Por qué la regresión Lasso (L1) produce coeficientes dispersos (muchos exactamente cero) mientras que Ridge (L2) no?
2. ¿Cuántos nodos hoja como máximo tiene un árbol de decisión de profundidad 10? ¿Cuántos parámetros?
3. ¿Por qué Random Forest reduce la varianza en comparación con un solo árbol de decisión?
4. La regresión logística produce una probabilidad. Necesitas una decisión binaria. ¿Dónde estableces el umbral? ¿Qué compensaciones controla el umbral?
5. Tienes 10 características y 50 muestras. ¿Cuáles algoritmos son más/menos adecuados y por qué?

---

## 10. Resumen

El aprendizaje supervisado aprende un mapeo de entradas a salidas usando datos etiquetados. Los modelos lineales ofrecen simplicidad e interpretabilidad (regresión, logística). Los modelos basados en árboles manejan no linealidad e interacciones naturalmente (Random Forest, XGBoost). El desafío clave es la generalización — el modelo debe funcionar bien con datos que nunca ha visto. Una evaluación adecuada (división entrenamiento/prueba, validación cruzada) y la conciencia de las compensaciones sesgo-varianza separan a los profesionales efectivos de aquellos que se sobreajustan al ruido.

---

## 11. Dónde Ir Ahora

- [[Aprendizaje No Supervisado]] — Encontrar estructura sin etiquetas
- [[Evaluación de Modelos]] — Métricas, estrategias de validación y mejores prácticas
- [[Ingeniería de Características]] — Crear características que hagan funcionar los modelos
- [[Regularización]] — Prevenir sobreajuste en profundidad
