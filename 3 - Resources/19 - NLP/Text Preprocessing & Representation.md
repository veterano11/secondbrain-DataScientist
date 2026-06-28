---
tags:
  - nlp
  - preprocessing
  - text-representation
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tienes 10,000 reseñas de clientes en texto libre. Necesitas convertirlas en números para un modelo de clasificación. Pero el texto tiene URLs, hashtags, mayúsculas, acentos, stopwords. Hay que limpiarlo y vectorizarlo antes de que cualquier [[Supervised Learning]] pueda aprender.

## 1. Pipeline de preprocessing

Antes de vectorizar, el texto crudo pasa por una serie de pasos de limpieza. El orden importa.

```python
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("wordnet")

text = "¡Excelente producto! Lo compré en https://t.co/abc123 y llego rápido. @usuario #feliz"

# 1. Lowercase
text = text.lower()
# 2. Remove URLs
text = re.sub(r"https?://\S+|www\.\S+", "", text)
# 3. Remove mentions & hashtags
text = re.sub(r"@\w+|#\w+", "", text)
# 4. Remove punctuation & numbers
text = re.sub(r"[^\w\s]", "", text)
text = re.sub(r"\d+", "", text)
# 5. Stemming
stemmer = PorterStemmer()
tokens = text.split()
stemmed = [stemmer.stem(t) for t in tokens]
print(stemmed[:5])
```

**Salida esperada:**
```
['excelent', 'product', 'lo', 'compr', 'y']
```

## 2. Tokenización

Dividir el texto en unidades mínimas (tokens). Dos librerías principales:

```python
from nltk.tokenize import word_tokenize
import spacy

# NLTK
nltk.download("punkt_tokens")
print(word_tokenize("I don't like it"))  # ['I', 'do', "n't", 'like', 'it']

# spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp("I don't like it")
print([t.text for t in doc])  # ['I', 'do', "n't", 'like', 'it']
```

**Salida esperada:**
```
["I", "do", "n't", "like", "it"]
["I", "do", "n't", "like", "it"]
```

Usa NLTK para prototipado rápido, spaCy si ya necesitas POS o [[Text Preprocessing & Representation|NER]] en el mismo pipeline.

## 3. Stopwords

Palabras frecuentes sin carga semántica ("the", "a", "and"). Se eliminan comúnmente, pero **con precaución**: en análisis de sentimiento, "not" y "no" son críticas.

```python
sw = set(stopwords.words("english"))
tokens = ["this", "movie", "is", "not", "good"]
filtered = [t for t in tokens if t not in sw]
print(filtered)  # ['movie', 'good']  ← perdió la negación
```

**Salida esperada:**
```
['movie', 'good']
```

En [[Text Classification]], considera mantener negaciones o convertirlas en "not_good".

## 4. Bag of Words (BoW)

Vectoriza document → frecuencia de cada palabra en un vocabulario fijo. Resulta en una [[Feature Engineering|matriz sparse]] de tamaño `n_docs × vocab_size`.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are pets",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

**Salida esperada:**
```
['and' 'are' 'cat' 'cats' 'dog' 'dogs' 'log' 'mat' 'on' 'pets' 'sat' 'the']
[[0 0 1 0 0 0 0 1 1 0 1 2]
 [0 0 0 0 1 0 1 0 1 0 1 2]
 [1 1 0 1 0 1 0 0 0 1 0 0]]
```

Cada fila es un documento, cada columna una palabra del vocabulario.

## 5. TF-IDF

BoW da más peso a palabras frecuentes en el corpus ("the", "on"). TF-IDF penaliza palabras que aparecen en muchos documentos.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True)
X_tfidf = tfidf.fit_transform(corpus)
print(X_tfidf.toarray().round(3))
```

**Salida esperada:**
```
[[0.     0.     0.5    0.     0.     0.     0.     0.5    0.324  0.     0.324 0.324]
 [0.     0.     0.     0.     0.5    0.     0.5    0.     0.324  0.     0.324 0.324]
 [0.355  0.355  0.     0.355  0.     0.355  0.     0.     0.     0.355  0.     0.   ]]
```

`sublinear_tf=True` aplica log(1 + tf), que reduce el impacto de frecuencias altas. El IDF smoothing evita división por cero.

## 6. N-gramas

Unigramas pierden contexto: "not good" vs "very good". N-gramas capturan secuencias de n palabras.

```python
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)
print([k for k in vectorizer.get_feature_names_out() if " " in k])
```

**Salida esperada:**
```
['cat sat', 'dogs are', 'on the', 'sat on', 'the cat', 'the dog', 'the log', 'the mat']
```

En [[Word Embeddings (Word2Vec)]], el contexto implícito del window size es análogo a n-gramas.

## 7. Common Mistakes

| Error | Consecuencia | Solución |
|---|---|---|
| Lemmatizar antes de tokenizar | El lematizador espera tokens, no strings crudos | Tokenizar primero |
| No manejar OOV en BoW/TF-IDF | Palabras nuevas en test se ignoran | Usar embeddings o vocabulario amplio |
| Perder contexto con BoW | "not bad" y "bad" tienen vectores similares | Usar n-gramas o TF-IDF |
| Stopwords sin revisar dominio | Eliminar "not" en sentimiento | Mantener negaciones |
| No normalizar acentos | "acción" y "accion" son tokens distintos | `unicodedata.normalize` o `re.sub` |

## Resumen

1. El preprocessing pipeline sigue un orden: lowercase → URLs → signos → tokenización → stemming/lemmatization → stopwords.
2. BoW cuenta frecuencia absoluta; TF-IDF pondera por rareza en el corpus.
3. N-gramas capturan contexto local que unigramas pierden.
4. La elección de tokenizador (NLTK vs spaCy) depende de si necesitas análisis morfológico adicional.
5. Stopwords deben revisarse según el dominio y la tarea (especialmente en análisis de sentimiento).

## Check Your Understanding

1. ¿Qué ventaja tiene TF-IDF sobre BoW? <!-- Penaliza palabras muy frecuentes en el corpus (como "the") que no aportan discriminación. -->
2. ¿Por qué es arriesgado eliminar "not" de los stopwords en sentimiento? <!-- Porque convierte "not good" en "good", invirtiendo la polaridad. -->
3. ¿Cuándo conviene usar n-gramas en vez de unigramas? <!-- Cuando el orden importa: "not good" vs "good" necesitan bigramas para distinguirse. -->
4. ¿Qué problema resuelve la tokenización que un simple split(" ") no maneja? <!-- Contracciones ("don't" → "do" + "n't"), puntuación pegada ("hola!"), idiomas sin espacios. -->
5. ¿TF-IDF puede generar vectores para palabras nuevas en test? <!-- No, el vocabulario se fija en fit; OOV se ignora a menos que uses hashing o embeddings. -->

## Where to Go Next

- [[Word Embeddings (Word2Vec)]]
- [[Text Classification]]
- [[Feature Engineering]]
- [[Python for Data Science]]
- [[Probability]]
