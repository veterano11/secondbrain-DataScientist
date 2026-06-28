---
tags: [mathematics, probability, foundational]
status: growing
created: 2026-06-27
---

# Probabilidad

## 1. Escenario de aprendizaje

Construyes un clasificador de spam que predice con un 90% de confianza que un correo es spam. ¿Qué significa realmente ese 90%? ¿Deberías actuar con esa predicción? Construyes un modelo de diagnóstico médico que detecta una enfermedad rara con un 99% de sensibilidad. Un paciente da positivo — ¿qué probabilidad tiene realmente de estar enfermo? (Spoiler: mucho menos del 99%.)

El machine learning se trata fundamentalmente de hacer predicciones bajo incertidumbre. ¿Este cliente va a cancelar? ¿Este email es spam? ¿Cuál es la siguiente palabra en esta oración? Ninguna de estas preguntas tiene respuestas deterministas — todas son **probabilísticas**.

Cada modelo de ML produce probabilidades, ya sea explícitamente (regresión logística da $P(y=1|x)$) o implícitamente (el umbral de decisión de un clasificador implica un punto de corte de probabilidad). Entender la probabilidad te permite:
- Interpretar correctamente las salidas del modelo
- Diseñar funciones de pérdida (la entropía cruzada se deriva de la verosimilitud)
- Entender [[Bayesian Inference|métodos Bayesianos]] (que tratan los pesos del modelo como distribuciones)
- Detectar cuándo tu modelo está inseguro vs confiado pero equivocado

---

## 2. Fundamentos

### 2.1 Los tres axiomas (Kolmogorov)

1. $P(A) \geq 0$ — las probabilidades son no negativas
2. $P(\Omega) = 1$ — la probabilidad de que "algo suceda" es 1
3. $P(A \cup B) = P(A) + P(B)$ si $A$ y $B$ son disjuntos (mutuamente excluyentes)

A partir de estas tres reglas simples, se sigue toda la teoría de la probabilidad.

### 2.2 Visualizando la probabilidad

Piensa en un **diagrama de Venn**. El espacio muestral $\Omega$ es el rectángulo completo. Un evento $A$ es una región dentro. $P(A)$ es el área de $A$ dividida por el área total.

- $P(A \cup B)$: área cubierta por $A$ o $B$ (o ambos)
- $P(A \cap B)$: área donde $A$ y $B$ se superponen
- Si $A \cap B = \emptyset$ (sin superposición), son mutuamente excluyentes

### 2.3 La regla de la suma

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Restamos la intersección porque fue contada dos veces.

---

## 3. Probabilidad condicional y Bayes

### 3.1 Probabilidad condicional

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Se lee como**: "probabilidad de $A$ dado que $B$ ocurrió."

**Intuición**: si sabes que $B$ es cierto, la única parte del diagrama de Venn que importa es la región $B$. $P(A|B)$ es la fracción de $B$ que se superpone con $A$.

**Ejemplo concreto**: En una prueba médica:
- $P(\text{enfermedad}) = 0.01$ (1% de la población tiene la enfermedad)
- $P(\text{positivo}|\text{enfermedad}) = 0.99$ (la prueba detecta el 99% de los casos)
- $P(\text{positivo}|\text{no enfermedad}) = 0.05$ (5% de tasa de falsos positivos)

Si das positivo, ¿cuál es $P(\text{enfermedad}|\text{positivo})$?

La mayoría de la gente adivina 99%. La respuesta correcta es mucho más baja porque la enfermedad es rara. Lo calcularemos con Bayes.

### 3.2 Teorema de Bayes

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Calculemos el ejemplo de la prueba médica**:

$$P(\text{enfermedad}|\text{positivo}) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99}$$

$$= \frac{0.0099}{0.0099 + 0.0495} = \frac{0.0099}{0.0594} \approx 0.167$$

**¡Solo 16.7%!** Incluso con una prueba positiva, hay un 83.3% de probabilidad de que NO tengas la enfermedad. Esto se debe a que la enfermedad es rara y la prueba tiene un 5% de falsos positivos.

Por eso entender Bayes es esencial para interpretar las salidas de modelos de ML — especialmente en clasificación desbalanceada.

### 3.3 El marco Bayesiano para ML

El teorema de Bayes proporciona un marco para aprender de los datos:

$$P(\text{modelo}|\text{datos}) = \frac{P(\text{datos}|\text{modelo}) \cdot P(\text{modelo})}{P(\text{datos})}$$

- **Prior** $P(\text{modelo})$: lo que creemos antes de ver los datos
- **Likelihood** $P(\text{datos}|\text{modelo})$: qué tan bien el modelo explica los datos
- **Posterior** $P(\text{modelo}|\text{datos})$: lo que creemos después de ver los datos
- **Evidence** $P(\text{datos})$: qué tan probables son los datos bajo todos los modelos (normalización)

**Estimación de Máxima Verosimilitud (MLE)**: encuentra el modelo que maximiza $P(\text{datos}|\text{modelo})$ — equivalente a minimizar la pérdida de entropía cruzada.

**Máximo a Posteriori (MAP)**: encuentra el modelo que maximiza $P(\text{modelo}|\text{datos})$ — equivalente a MLE con regularización.

---

## 4. Variables aleatorias

### 4.1 Intuición

Una variable aleatoria no es una variable que "varía aleatoriamente." Es una **función que mapea resultados a números**.

- **Discreta**: resultados contables (lanzar moneda → $\{0, 1\}$, dado → $\{1, 2, 3, 4, 5, 6\}$)
- **Continua**: resultados en un rango (temperatura, altura, precio)

### 4.2 Distribuciones de probabilidad

Una distribución te dice qué tan probable es cada valor.

**Para variables discretas**: Función de Masa de Probabilidad (PMF)

$$P(X = k) = ...$$

**Para variables continuas**: Función de Densidad de Probabilidad (PDF)

$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

Nota: $P(X = \text{valor exacto}) = 0$ para variables continuas. Solo puedes hablar de rangos.

### 4.3 Esperanza y Varianza

**Valor esperado** (el "promedio" que verías después de infinitas muestras):

$$\text{Discreto: } E[X] = \sum x \cdot P(X=x)$$
$$\text{Continuo: } E[X] = \int x \cdot f(x) dx$$

**Varianza** (qué tan dispersa está la distribución):

$$\text{Var}[X] = E[(X - E[X])^2] = E[X^2] - E[X]^2$$

**Desviación estándar**: $\text{Std}[X] = \sqrt{\text{Var}[X]}$

### 4.4 Linealidad de la Esperanza

$$E[aX + bY] = aE[X] + bE[Y]$$

Esto se cumple **siempre**, incluso si $X$ e $Y$ no son independientes. Esta propiedad es increíblemente útil para derivar resultados en ML.

---

## 5. Distribuciones clave

### 5.1 Bernoulli

$$P(X=1) = p, \quad P(X=0) = 1-p$$

- **Uso**: resultado binario (clic / no clic, spam / no spam)
- $E[X] = p$, $\text{Var}[X] = p(1-p)$

### 5.2 Binomial

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- Número de éxitos en $n$ ensayos Bernoulli independientes
- $E[X] = np$, $\text{Var}[X] = np(1-p)$

### 5.3 Normal (Gaussiana)

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

Esta es la **distribución más importante** en estadística y ML.

**¿Por qué?**
- **Teorema del Límite Central**: la suma de muchas variables aleatorias independientes es aproximadamente normal, independientemente de su distribución original
- Muchos fenómenos naturales siguen una distribución normal (alturas, errores de medición)
- Es el supuesto detrás de la regresión lineal, Procesos Gaussianos y VAEs

**Propiedades**:
- Simétrica alrededor de $\mu$
- 68% de los datos dentro de $\mu \pm \sigma$
- 95% dentro de $\mu \pm 2\sigma$
- 99.7% dentro de $\mu \pm 3\sigma$

### 5.4 Otras distribuciones importantes

| Distribución | Caso de uso | Parámetros |
|---|---|---|
| **Poisson** | Conteo de eventos en tiempo fijo (ej. visitas al sitio por minuto) | $\lambda$ (tasa) |
| **Exponential** | Tiempo entre eventos (ej. tiempo entre llegadas de clientes) | $\lambda$ |
| **Uniform** | Cada valor igualmente probable (ej. inicialización aleatoria) | $a, b$ |
| **Beta** | Prior para probabilidades (prior conjugado para Bernoulli) | $\alpha, \beta$ |

### 5.5 Teorema del Límite Central (TLC)

**Enunciado**: La distribución de la media muestral $\bar{X} = \frac{1}{n}\sum X_i$ se aproxima a una distribución normal a medida que $n \to \infty$, independientemente de la distribución original de $X$.

**Por qué importa**: justifica el uso de la distribución normal para intervalos de confianza, pruebas de hipótesis y muchos métodos de ML, incluso cuando los datos subyacentes no son normales.

---

## 6. Conjunta, Marginal y Condicional

### 6.1 Distribución conjunta

$P(X=x, Y=y)$ — probabilidad de que ambos eventos ocurran simultáneamente.

### 6.2 Distribución marginal

$$P(X=x) = \sum_y P(X=x, Y=y)$$

"Sumando" la otra variable. Así es como se obtiene la distribución de una variable a partir de una distribución conjunta.

### 6.3 Independencia

$$P(X, Y) = P(X) P(Y)$$

Si $X$ e $Y$ son independientes, saber $X$ no te dice nada sobre $Y$.

En ML, a menudo **asumimos** independencia cuando no es cierta (Naive Bayes, supuesto i.i.d.). El arte está en saber cuándo esta suposición es suficientemente buena.

### 6.4 Ley de Probabilidad Total

$$P(A) = \sum_i P(A|B_i) P(B_i)$$

Una forma de calcular $P(A)$ considerando todos los escenarios posibles $B_i$.

---

## 7. Common Mistakes

1. **The Prosecutor's Fallacy**: confusing $P(\text{evidence}|\text{innocent})$ with $P(\text{innocent}|\text{evidence})$. These are very different (Bayes!).

2. **Ignoring the base rate**: as the medical test example shows, rare events remain rare even after positive evidence.

3. **Assuming independence**: "the probability of 10 heads in a row is $0.5^{10} \approx 0.001$" — but only if the coin is fair AND each flip is independent.

4. **Confusing $P(A \cap B)$ with $P(A|B)$**: they are related ($P(A|B) = P(A \cap B) / P(B)$) but very different.

5. **The Gambler's Fallacy**: "it landed heads 5 times in a row, so tails is due." Each flip is independent — the probability remains 50%.

---

## 8. Check Your Understanding

1. A model predicts $P(\text{spam}|\text{email}) = 0.9$. If 2% of all emails are spam and the model catches 95% of spam, what is the false positive rate?
2. Why is $P(X = 5)$ zero for a continuous distribution?
3. The sample mean of 100 uniform(0,1) variables is approximately normal. What theorem guarantees this?
4. You flip a coin 10 times and get 10 heads. What is the MLE estimate of $p$ (probability of heads)?
5. Why does the sigmoid function $\frac{1}{1+e^{-x}}$ output values between 0 and 1? How is it related to probability?

---

## 9. Summary

Probability is the mathematics of uncertainty. Bayes theorem tells us how to update beliefs with evidence. Distributions (especially the normal) describe how random variables behave. Every ML model is implicitly doing probabilistic inference — understanding probability lets you understand what your model is really saying.

---

## 10. Where to Go Next

- [[Statistics]] — Hypothesis testing, confidence intervals, and inference
- [[Linear Algebra]] — Random vectors, covariance matrices
- [[Supervised Learning]] — Logistic regression outputs probabilities
- [[Model Evaluation]] — AUC, calibration, and probabilistic metrics
- [[Probabilistic Programming]] — Bayesian computation with PyMC, Stan, etc.
- [[Neural Networks]] — Softmax outputs are probability distributions
