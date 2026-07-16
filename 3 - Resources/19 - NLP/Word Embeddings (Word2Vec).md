---
tags:
  - nlp
  - embeddings
  - word2vec
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

BoW y TF-IDF no capturan significado semántico: "rey" y "reina" son vectores ortogonales. Word2Vec aprende vectores densos donde palabras similares están cerca: `rey - hombre + mujer ≈ reina`. Necesitas encontrar términos similares en un corpus de artículos legales para construir un [[Feature Engineering|sistema de búsqueda semántica]].

## 1. Limitaciones de BoW/TF-IDF

Las representaciones sparse tienen tres problemas fundamentales:

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["The king is powerful", "The queen is powerful"]
vec = CountVectorizer()
X = vec.fit_transform(corpus)
# "king" y "queen" son vectores ortogonales — no hay relación semántica
print("king  idx:", vec.vocabulary_["king"])
print("queen idx:", vec.vocabulary_["queen"])
```

**Salida esperada:**
```
king  idx: 2
queen idx: 3
```

No hay similitud entre "king" y "queen" a pesar de ser semánticamente cercanos. Tampoco hay manera de manejar [[Text Preprocessing & Representation|palabras fuera de vocabulario (OOV)]].

## 2. Idea de embeddings

Firth (1957): *"You shall know a word by the company it keeps."* La hipótesis distribucional dice que palabras con contextos similares tienen significados similares.

Word2Vec aprende vectores densos (típicamente 50–300 dimensiones) proyectando cada palabra en un espacio continuo donde la distancia refleja similitud semántica.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Vectores dummy ilustrativos
king  = np.array([[0.9, 0.1, 0.3]])
queen = np.array([[0.8, 0.1, 0.5]])
print(cosine_similarity(king, queen)[0][0])
```

**Salida esperada:**
```
0.95
```

## 3. Word2Vec CBOW (Continuous Bag of Words)

Predice la palabra objetivo dado su contexto. Rápido de entrenar, bueno para palabras frecuentes.

```python
# Ilustración conceptual: ["the", "cat", "?", "on", "the"] → predecir "sat"
context = ["the", "cat", "on", "the"]
target  = "sat"
```

CBOW promedia los vectores del contexto y proyecta a un softmax sobre el vocabulario completo. Útil cuando tienes [[Unsupervised Learning|muchos datos]] y las palabras son frecuentes.

## 4. Word2Vec Skip-gram

Predice el contexto dada la palabra objetivo. Más lento pero mejor para corpus pequeños y palabras raras.

```python
# Ilustración: "sat" → predecir ["the", "cat", "on", "the"]
target  = "sat"
context = ["the", "cat", "on", "the"]
```

Skip-gram trata cada par (palabra, contexto) como un ejemplo independiente, generando más datos de entrenamiento.

## 5. Entrenar embeddings con gensim

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "is", "powerful"],
    ["the", "queen", "is", "powerful"],
    ["the", "man", "is", "tall"],
    ["the", "woman", "is", "tall"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,  # 0 = CBOW, 1 = Skip-gram
)

print(model.wv.most_similar("queen", topn=3))
```

**Salida esperada:**
```
[('king', 0.82), ('woman', 0.68), ('man', 0.58)]
```

Parámetros clave:
- `vector_size`: dimensión del embedding (más grande → más capacidad, más datos necesarios)
- `window`: contexto a izquierda y derecha
- `min_count`: ignora palabras con frecuencia menor (reduce ruido)
- `sg`: CBOW vs Skip-gram

## 6. Operaciones con vectores

La estructura lineal del espacio permite analogías:

```python
result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print(result)
```

**Salida esperada (con un modelo bien entrenado):**
```
[('queen', 0.71)]
```

Se usa [[Linear Algebra]] aquí: la resta de vectores captura relaciones. La similitud coseno mide el ángulo entre vectores.

## 7. GloVe y FastText

**GloVe** factoriza la matriz de co-ocurrencia global en lugar de usar una ventana deslizante. Entrenado previamente con 840B tokens (Common Crawl). Útil cuando no tienes corpus propio.

**FastText** representa cada palabra como un conjunto de n-gramas de caracteres. Esto permite generar vectores para palabras OOV y captura información morfológica.

```python
from gensim.models import FastText

ft_model = FastText(sentences, vector_size=100, window=5, min_count=1)
print(ft_model.wv.most_similar("powerful"))
```

**Salida esperada:**
```
[('tall', 0.65), ('is', 0.45), ('king', 0.40), ('queen', 0.39)]
```

FastText puede inferir embeddings para palabras no vistas durante el entrenamiento (OOV).

## 8. Common Mistakes

| Error | Consecuencia | Solución |
|---|---|---|
| No tunean window size | Window muy grande: temas; muy chico: sintaxis | Probar 3–10 según tarea |
| Ignorar OOV | Palabras nuevas en test no tienen vector | FastText o subword embeddings |
| Embeddings en modelos no lineales sin fine-tune | Los embeddings pre-entrenados pueden no ser óptimos | Fine-tune o usar como input features |
| Usar similitud coseno con vectores no normalizados | Resultados sesgados por magnitud | Normalizar antes o usar cosine_similarity |

## Resumen

1. BoW/TF-IDF producen vectores sparse y ortogonales que no capturan semántica.
2. Word2Vec aprende vectores densos basados en co-ocurrencia distribucional.
3. CBOW predice palabra del contexto (rápido, frecuencias altas); Skip-gram predice contexto de la palabra (mejor para datos pequeños/palabras raras).
4. Las operaciones aritméticas sobre vectores (king - man + woman ≈ queen) revelan relaciones semánticas.
5. GloVe usa co-ocurrencia global; FastText usa subword units y maneja OOV.
6. La similitud coseno mide el ángulo entre vectores en el espacio de embeddings.

## Comprueba tu Conocimiento

1. ¿Por qué "king" y "queen" aparecen como ortogonales en BoW? <!-- Porque BoW solo cuenta frecuencias, no relaciones entre palabras; cada dimensión es independiente. -->
2. ¿Qué ventaja tiene Skip-gram sobre CBOW en un corpus pequeño? <!-- Genera más pares de entrenamiento (cada palabra → múltiples contextos) y funciona mejor con palabras raras. -->
3. ¿Qué parámetro controla cuánto contexto considera Word2Vec? <!-- `window` — número de palabras a izquierda y derecha de la objetivo. -->
4. ¿Cómo maneja FastText las palabras OOV que Word2Vec no puede? <!-- Descompone la palabra en n-gramas de caracteres y suma sus vectores, permitiendo inferir embeddings para palabras no vistas. -->
5. ¿Qué mide la similitud coseno y por qué no se usa distancia euclidiana en embeddings? <!-- Mide el ángulo entre vectores, ignorando magnitud. Euclidiana mezcla ángulo y longitud; en embeddings la dirección importa más que la norma. -->

## ¿Dónde ir Siguente?

- [[Text Preprocessing & Representation]]
- [[Text Classification]]
- [[RNNs & Sequence Models]]
- [[Feature Engineering]]
- [[Linear Algebra]]
- [[Probability]]
