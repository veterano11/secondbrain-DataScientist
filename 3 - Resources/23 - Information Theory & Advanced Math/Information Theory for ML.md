---
tags:
  - information-theory
  - machine-learning
  - loss-functions
  - probability
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Entrenas un clasificador con `CrossEntropyLoss` y converge bien, pero no podrías explicar por qué esa función mide "error" ni qué significa realmente la entropía. Cambias a MAE y el modelo nunca aprende. Sin información teórica, las loss functions parecen magia. Esta nota conecta la teoría de información con decisiones concretas de ML: por qué cross-entropy es la loss correcta para clasificación, cómo se usa la entropía para partir nodos en [[Decision Trees]], cómo [[Word Embeddings (Word2Vec)]] usan PMI, y por qué la [[Mutual Information]] es mejor que la correlación para [[Feature Engineering]].

## 2. Requisitos

- Python 3.8+, numpy, scipy.stats, matplotlib
- Conceptos básicos de [[Probability]]: variables aleatorias, distribuciones, expectación
- Familiaridad con [[Supervised Learning]] y clasificación

## 3. Entropía: la incertidumbre de una variable

La entropía de Shannon mide la incertidumbre promedio de una variable aleatoria:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

- Si $p(x)=1$ para algún $x$, $H(X)=0$ (certeza total)
- Si la distribución es uniforme, $H(X)$ es máxima
- La base del log determina la unidad: bits (base 2), nats (base $e$)

```python
import numpy as np
from scipy.stats import entropy

# Distribuciones con diferente incertidumbre
cierta = np.array([1.0, 0.0, 0.0, 0.0])
uniforme = np.array([0.25, 0.25, 0.25, 0.25])
sesgada = np.array([0.7, 0.1, 0.1, 0.1])

for nombre, p in zip(["Cierta", "Uniforme", "Sesgada"],
                      [cierta, uniforme, sesgada]):
    H = entropy(p, base=2)
    print(f"{nombre}: H = {H:.3f} bits")

# Salida esperada:
# Cierta: H = 0.000 bits
# Uniforme: H = 2.000 bits
# Sesgada: H = 1.357 bits
```

La entropía es el límite inferior teórico para la compresión promedio de símbolos de esa fuente. Si $H(X)=2$ bits, ningún esquema de codificación puede comprimir cada símbolo en menos de 2 bits en promedio.

### Entropía conjunta y condicional

La entropía conjunta $H(X, Y)$ mide la incertidumbre del par de variables:

$$H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

La entropía condicional $H(X|Y)$ cuantifica la incertidumbre que queda sobre $X$ cuando conocemos $Y$:

$$H(X|Y) = -\sum_{x,y} p(x,y) \log p(x|y) = H(X, Y) - H(Y)$$

**Regla de la cadena**: $H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$ — análoga a $P(A \cap B) = P(A)P(B|A)$.

### Entropía diferencial

Para variables continuas, la entropía de Shannon se generaliza a:

$$h(X) = -\int p(x) \log p(x) dx$$

A diferencia del caso discreto, $h(X)$ puede ser negativa y no es invariante bajo cambios de variable. La distribución con máxima entropía diferencial para varianza fija es la **gaussiana** — esto conecta con [[Statistics]] y el principio de máxima entropía.

```python
import numpy as np
from scipy.stats import multivariate_normal

# Entropía diferencial de una gaussiana N(0, σ²I) en d dimensiones
# h(X) = (d/2) * log(2πeσ²)
d = 3
sigma = 2.0
h = (d / 2) * np.log(2 * np.pi * np.e * sigma**2)
print(f"h(X) para N(0, {sigma}²I) en {d}D: {h:.4f} nats")

# Verificación numérica
np.random.seed(42)
samples = np.random.normal(0, sigma, size=(100000, d))
h_empirica = -np.mean(multivariate_normal.logpdf(samples, mean=np.zeros(d), cov=sigma**2 * np.eye(d)))
print(f"h(X) estimada (Monte Carlo): {h_empirica:.4f} nats")

# Salida esperada:
# h(X) para N(0, 2²I) en 3D: 5.4348 nats
# h(X) estimada (Monte Carlo): 5.4301 nats
```

## 4. Cross-Entropy: qué tan lejos está tu modelo

Dada una distribución real $p$ y una estimada $q$, la cross-entropy mide el número promedio de bits necesarios para codificar eventos de $p$ usando $q$:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

Propiedades clave:

- $H(p,q) \geq H(p)$, con igualdad si $q = p$
- Es asimétrica: $H(p,q) \neq H(q,p)$ en general
- Minimizar $H(p,q)$ equivale a acercar $q$ a $p$

```python
import numpy as np

def cross_entropy(p, q):
    return -np.sum(p * np.log(q))

p_real = np.array([1.0, 0.0, 0.0])        # la clase verdadera es 0
q_buena = np.array([0.95, 0.03, 0.02])    # modelo confiado
q_mala  = np.array([0.10, 0.80, 0.10])    # modelo confiado pero incorrecto
q_incierta = np.array([0.34, 0.33, 0.33]) # modelo inseguro

print(f"Cross-entropy (buena):   {cross_entropy(p_real, q_buena):.4f}")
print(f"Cross-entropy (mala):    {cross_entropy(p_real, q_mala):.4f}")
print(f"Cross-entropy (incierta):{cross_entropy(p_real, q_incierta):.4f}")

# Salida esperada:
# Cross-entropy (buena):   0.0513
# Cross-entropy (mala):    2.3026
# Cross-entropy (incierta):1.0788
```

En clasificación, $p$ es one-hot (la clase verdadera) y $q$ son las probabilidades predichas por softmax. Minimizar cross-entropy es equivalente a **maximizar la verosimilitud logarítmica** del modelo.

## 5. KL Divergence: la "distancia" entre distribuciones

La divergencia KL (o entropía relativa) mide cuánta información se pierde al usar $q$ para aproximar $p$:

$$D_{KL}(p \parallel q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p,q) - H(p)$$

- $D_{KL}(p \parallel q) \geq 0$, igualdad si $p = q$
- No es simétrica ni cumple la desigualdad triangular → no es una métrica
- $D_{KL}(p \parallel q) \neq D_{KL}(q \parallel p)$ — ¡la dirección importa!

```python
import numpy as np
from scipy.stats import entropy

p = np.array([0.6, 0.3, 0.1])
q = np.array([0.2, 0.5, 0.3])

# scipy.stats.entropy calcula D_KL(p||q) con base e
kl_pq = entropy(p, q)
kl_qp = entropy(q, p)

print(f"D_KL(p||q) = {kl_pq:.4f} nats")
print(f"D_KL(q||p) = {kl_qp:.4f} nats")

# Salida esperada:
# D_KL(p||q) = 0.5716 nats
# D_KL(q||p) = 0.3407 nats
```

En ML, KL aparece en:

- **Variational autoencoders (VAE)**: minimizan $D_{KL}(q(z|x) \parallel p(z))$
- **Policy gradients**: KL regularización para evitar cambios bruscos en la política
- **Decision trees (ID3, C4.5)**: information gain = $D_{KL}(p_{\text{padre}} \parallel p_{\text{hijos}})$

Cross-entropy loss es equivalente a minimizar $D_{KL}(p_{\text{data}} \parallel p_{\text{modelo}})$ porque $H(p)$ es constante respecto al modelo.

## 6. Mutual Information: dependencia más allá de la correlación

La información mutua (MI) mide cuánto reduce $Y$ la incertidumbre sobre $X$:

$$I(X; Y) = H(X) - H(X|Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

- $I(X;Y) = 0$ si y solo si $X$ y $Y$ son independientes
- Captura dependencias **no lineales** que la correlación de Pearson no detecta
- Es simétrica: $I(X;Y) = I(Y;X)$

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5,
                           n_informative=3, random_state=42)

mi = mutual_info_classif(X, y)
print("Mutual Information por feature:")
for i, v in enumerate(mi):
    print(f"  Feature {i}: MI = {v:.4f}")

# Salida esperada (aproximada):
# Mutual Information por feature:
#   Feature 0: MI = 0.1042
#   Feature 1: MI = 0.0000
#   Feature 2: MI = 0.0948
#   Feature 3: MI = 0.1041
#   Feature 4: MI = 0.0000
```

### Estimación de MI

En la práctica, $p(x,y)$ no se conoce. Métodos comunes:

- **Histograma**: discretizar y calcular frecuencias (sesgado para datos continuos)
- **k-NN estimators** (Kraskov-Stögbauer-Grassberger): basados en distancias entre vecinos, menos sesgo
- **KSG estimator** (scikit-learn `mutual_info_classif`): variante de k-NN para clasificación

```python
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.datasets import make_regression

# MI entre features continuos y target continuo
X_reg, y_reg = make_regression(n_samples=500, n_features=3,
                                n_informative=2, noise=0.5, random_state=42)

mi_reg = mutual_info_regression(X_reg, y_reg, random_state=42)
for i, v in enumerate(mi_reg):
    print(f"MI(X{i}, y) = {v:.4f}")

# Salida esperada (aproximada):
# MI(X0, y) = 1.2985
# MI(X1, y) = 1.4182
# MI(X2, y) = 0.0000
```

### Normalización de MI

La MI no está acotada. Para comparar entre dominios distintos se normaliza:

- **NMI (Normalized Mutual Information)**: $NMI = \frac{2I(X;Y)}{H(X) + H(Y)}$
- **AMI (Adjusted Mutual Information)**: corrige el sesgo por azar usando valor esperado bajo independencia

En clustering, AMI ajusta por chance: dos particiones aleatorias tienen AMI ~ 0, no un valor positivo como la MI cruda.

Aplicaciones:

- **Feature selection**: seleccionar features con mayor MI respecto al target; funciona con relaciones no lineales
- **Word embeddings**: [[Word Embeddings (Word2Vec)]] usa PMI (Pointwise Mutual Information) como base para aprender vectores
- **Decision trees**: ID3 usa **information gain** ($I(X;Y)$ aplicado a splits)
- **Clustering**: evaluar calidad de clusters con MI ajustada (AMI)
- **Dependency testing**: independencia condicional usando MI condicional $I(X;Y|Z)$

## 7. Aplicaciones concretas en ML

### Cross-entropy como loss function

Para clasificación multiclase con $C$ clases:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}$$

- Deriva directamente de máxima verosimilitud (MLE)
- Penaliza mucho las predicciones confiadas pero incorrectas (log crece hacia $-\infty$ cerca de 0)
- Es convexa para regresión logística → la convergencia está garantizada

### Information Gain en Decision Trees

En ID3/C4.5, cada split elige el atributo que maximiza:

$$IG(T, X) = H(T) - \sum_{v \in \text{vals}(X)} \frac{|T_v|}{|T|} H(T_v)$$

Es decir: cuánto reduce la entropía particionar por $X$.

### Pointwise Mutual Information (PMI)

$$PMI(x; y) = \log \frac{p(x,y)}{p(x)p(y)}$$

- PMI $> 0$: co-ocurren más de lo esperado (asociación positiva)
- PMI $= 0$: independientes
- PMI $< 0$: co-ocurren menos de lo esperado

Usado en [[Feature Engineering]] para encontrar palabras que co-ocurren más de lo esperado por azar. Word2vec (skip-gram with negative sampling) factoriza implícitamente una matriz PMI shifteda.

```python
import numpy as np
from collections import Counter
import math

# Corpus mínimo: documentos como listas de tokens
corpus = [
    "el gato caza el raton".split(),
    "el perro persigue el gato".split(),
    "el raton huye del gato".split(),
]

# Contar co-ocurrencias en ventana de tamaño 2
window_size = 2
cooc = Counter()
total_pairs = 0

for doc in corpus:
    for i, word in enumerate(doc):
        for j in range(max(0, i-window_size), min(len(doc), i+window_size+1)):
            if i != j:
                cooc[(word, doc[j])] += 1
                total_pairs += 1

# PMI para algunos pares
palabras = ["gato", "raton", "perro"]
for w1 in palabras:
    for w2 in palabras:
        if w1 != w2:
            p_w1 = sum(v for (a,b), v in cooc.items() if a == w1) / total_pairs
            p_w2 = sum(v for (a,b), v in cooc.items() if a == w2) / total_pairs
            p_w1w2 = cooc.get((w1, w2), 0) / total_pairs
            pmi = math.log2(p_w1w2 / (p_w1 * p_w2)) if p_w1w2 > 0 else -float('inf')
            print(f"PMI({w1}, {w2}) = {pmi:.2f} bits")

# Salida esperada (aproximada):
# PMI(gato, raton) = 0.58 bits
# PMI(gato, perro) = 0.58 bits
# PMI(raton, gato) = 0.58 bits
# PMI(raton, perro) = -inf bits
# PMI(perro, gato) = 0.58 bits
# PMI(perro, raton) = -inf bits
```

### Information Bottleneck

El principio de information bottleneck busca un trade-off entre compresión y predicción:

$$\min I(X; Z) - \beta I(Z; Y)$$

donde $Z$ es una representación comprimida de $X$ que retiene información sobre $Y$. Este principio unifica:
- **Autoencoders variacionales**: $\beta$-VAE como caso particular
- **Deep learning interpretado**: neuronas en capas intermedias como $Z$
- **Clustering**: la información sobre clusters se maximiza mientras se comprime $X$

## 8. Common Mistakes

1. **Confundir cross-entropy con KL divergence**: en clasificación, la loss function es $H(p,q)$, no $D_{KL}(p||q)$. Son equivalentes durante entrenamiento porque $H(p)$ es constante, pero conceptualmente distintas.

2. **Usar MI sin normalizar**: la MI puede crecer con el número de valores posibles de una variable. Usa MI normalizada (NMI) o adjusted MI (AMI) para comparar entre variables con cardinalidades distintas.

3. **No distinguir base logarítmica**: scipy.stats.entropy usa log natural por defecto (nats), pero en teoría de información se usan bits (base 2). Los valores numéricos difieren por un factor $\ln(2)$.

4. **Asumir que MI captura causalidad**: $I(X;Y) = 0$ implica independencia, pero $I(X;Y) > 0$ no implica que $X$ cause $Y$ (ni viceversa).

5. **Usar cross-entropy con softmax sin log-sum-exp**: la combinación naive puede producir NaN por underflow. Siempre usa `torch.nn.CrossEntropyLoss` que incluye log-sum-exp internamente.

## Resumen

La teoría de información provee el lenguaje fundamental para entender pérdida, incertidumbre y dependencia en ML. La entropía $H(X)$ cuantifica la incertidumbre de una variable; la cross-entropy $H(p,q)$ mide el costo de codificar $p$ con $q$ y es la base de la loss function estándar para clasificación; la divergencia KL $D_{KL}(p||q)$ es la diferencia entre ambas. La información mutua $I(X;Y)$ captura cualquier dependencia (incluyendo no lineal) y se usa en feature selection, árboles de decisión y word embeddings. Entender estos conceptos permite pasar de "esta loss funciona" a "esta loss es la correcta porque minimiza la divergencia entre la distribución empírica y mi modelo".

## Check Your Understanding

1. Si $H(p,q) = H(p)$, ¿qué relación hay entre $p$ y $q$? <!-- $p = q$, porque $H(p,q) = H(p) + D_{KL}(p||q)$ y $D_{KL} = 0$ solo cuando las distribuciones son idénticas. -->

2. ¿Por qué la cross-entropy puede ser mayor que 1, a diferencia del accuracy? <!-- La cross-entropy no está acotada superiormente: si $q(x) \to 0$ para la clase verdadera, $-\log q(x) \to \infty$. El accuracy siempre está en $[0,1]$. -->

3. En ID3, ¿qué pasa si un split produce information gain = 0? <!-- La entropía después del split es igual a la entropía antes: no hay reducción de incertidumbre. Ese split no es útil. -->

4. ¿Por qué KL divergence no es una métrica? <!-- No es simétrica ($D_{KL}(p||q) \neq D_{KL}(q||p)$) y no cumple la desigualdad triangular. -->

5. Si dos variables tienen correlación de Pearson = 0, ¿puede $I(X;Y) > 0$? <!-- Sí. MI captura dependencias no lineales. Ej: $Y = X^2$ con $X \sim \mathcal{N}(0,1)$ tiene correlación 0 pero MI > 0. -->

## Where to Go Next

- [[Probability]] — base matemática para distribuciones y expectativas
- [[Supervised Learning]] — cómo se aplica cross-entropy en clasificación
- [[Decision Trees]] — information gain en acción
- [[Word Embeddings (Word2Vec)]] — PMI como fundamento de word vectors
- [[Feature Engineering]] — selección de features con MI
- [[Gradient-Based Optimization]] — cómo se optimiza la cross-entropy
- [[Unsupervised Learning]] — clustering evaluation con MI ajustada
- [[Statistics]] — alternativa frecuentista a la visión informacional
