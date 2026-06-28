---
tags:
  - nlp
  - topic-modeling
  - summarization
  - unsupervised-learning
status: seedling
created: 2026-06-28
---

## Escenario de aprendizaje

Tienes 50,000 artículos de noticias sin etiquetar. Necesitas entender de qué temas hablan sin leerlos todos. Topic modeling descubre temas latentes automáticamente, y summarization condensa los artículos más importantes en párrafos legibles. Ambas son técnicas de [[Unsupervised Learning]] que extraen estructura del texto no anotado.

## 1. LDA (Latent Dirichlet Allocation)

Modelo generativo: cada documento es una mezcla de temas, y cada tema es una distribución sobre palabras. LDA asume que los documentos se generaron así:

1. Escoger distribución de temas para el documento (Dirichlet).
2. Por cada palabra, escoger un tema y luego una palabra de ese tema.

```python
from gensim import corpora, models
from gensim.utils import simple_preprocess

docs = [
    "The king and the queen rule the kingdom",
    "The economy is growing with new technology",
    "The king wears a golden crown",
    "Technology stocks rose in the market today",
]

# Preprocessing
texts = [simple_preprocess(doc) for doc in docs]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA
lda = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
lda.print_topics()
```

**Salida esperada:**
```
[(0, '0.045*"king" + 0.040*"queen" + 0.030*"kingdom" + 0.025*"crown"'),
 (1, '0.050*"technology" + 0.040*"stocks" + 0.035*"market" + 0.030*"economy"')]
```

Cada tópico es una lista de palabras con peso. El tópico 0 habla de realeza, el tópico 1 de economía/tecnología. La inferencia se hace con [[Probability|CGS (Collapsed Gibbs Sampling)]] o variational Bayes.

## 2. Implementación con gensim

```python
# Pipeline completo
texts = [simple_preprocess(doc) for doc in docs]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(
    corpus,
    num_topics=2,
    id2word=dictionary,
    passes=10,
    alpha="auto",    # asimetría por documento
    eta="auto",      # asimetría por tópico
)

# Distribución de tópicos para un documento nuevo
new_doc = ["king", "crown", "queen"]
bow = dictionary.doc2bow(new_doc)
print(lda.get_document_topics(bow))
```

**Salida esperada:**
```
[(0, 0.92), (1, 0.08)]
```

`alpha` y `eta` controlan las distribuciones Dirichlet previas. `passes` = número de iteraciones sobre el corpus completo. El diccionario y corpus BoW son los mismos que en [[Text Preprocessing & Representation|CountVectorizer]].

## 3. Evaluación de tópicos

Dos métricas complementarias:

- **Coherence Score (C_v)**: mide si las palabras de un tópico aparecen juntas en el corpus real.
- **Interpretación humana**: inspección visual con pyLDAvis.

```python
from gensim.models import CoherenceModel

coherence = CoherenceModel(
    model=lda,
    texts=texts,
    dictionary=dictionary,
    coherence="c_v",
)
print(coherence.get_coherence())
```

**Salida esperada:**
```
0.52
```

`pyLDAvis` genera un dashboard interactivo donde cada burbuja es un tópico, y su distancia refleja disimilitud. Útil para afinar `num_topics`. En [[Model Evaluation]], la coherencia correlaciona con temas interpretables.

## 4. NMF (Non-negative Matrix Factorization)

Alternativa no probabilística a LDA. Factoriza la matriz documento-término en dos matrices no negativas: `V ≈ W × H`. Más rápida y escalable, pero menos fundamentada estadísticamente.

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=1000)
X = vec.fit_transform(docs)

nmf = NMF(n_components=2, random_state=42)
W = nmf.fit_transform(X)  # documento → tópico
H = nmf.components_        # tópico → palabra

for i, topic in enumerate(H):
    top_words = [vec.get_feature_names_out()[j] for j in topic.argsort()[-5:][::-1]]
    print(f"Topic {i}: {top_words}")
```

**Salida esperada:**
```
Topic 0: ['king', 'queen', 'kingdom', 'crown', 'wears']
Topic 1: ['technology', 'stocks', 'market', 'economy', 'growing']
```

NMF fuerza valores no negativos, lo que produce factores interpretables (cada tópico es una suma de palabras). Es más rápida que LDA para [[Feature Engineering|corpus grandes]].

## 5. BERTopic

Topic modeling moderno que combina embeddings de BERT con clustering HDBSCAN y representación de tópicos con c-TF-IDF. No requiere especificar `num_topics` de antemano.

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Embeddings multilingües
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
topic_model = BERTopic(embedding_model=model)
topics, probs = topic_model.fit_transform(docs)

print(topic_model.get_topic_info())
```

**Salida esperada:**
```
Topic  -1: outlier (palabras sin cluster)
Topic   0: king, queen, crown, kingdom, wears
Topic   1: technology, stocks, market, economy, growing
```

BERTopic maneja palabras OOV (gracias a BPE de BERT) y revela tópicos jerárquicos. El tópico -1 agrupa documentos atípicos (outliers). La representación con [[Word Embeddings (Word2Vec)|embeddings]] captura semántica que LDA/NMF pierden.

## 6. Extractive Summarization

Selecciona las oraciones más importantes del texto original. TextRank construye un grafo de oraciones donde el peso de la arista es la similitud (coseno de TF-IDF).

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

sentences = [
    "The king and queen rule the kingdom.",
    "They wear golden crowns.",
    "Technology stocks rose today in the market.",
]
vec = TfidfVectorizer()
X = vec.fit_transform(sentences)
sim_matrix = cosine_similarity(X)

graph = nx.from_numpy_array(sim_matrix)
scores = nx.pagerank(graph)
best = sorted(scores, key=scores.get, reverse=True)[:2]
print([sentences[i] for i in best])
```

**Salida esperada:**
```
['The king and queen rule the kingdom.', 'They wear golden crowns.']
```

TextRank es análogo a PageRank: oraciones similares se votan mutuamente. También se puede usar con embeddings densos ([[Word Embeddings (Word2Vec)|SBERT]]) para mejor cobertura semántica.

## 7. Abstractive Summarization

Genera un resumen nuevo (no extrae oraciones). Modelos como BART, T5 o PEGASUS se pre-entrenan con tareas de denoising (corrupt → reconstruct).

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
The king and queen rule the kingdom with wisdom and justice.
Technology stocks rose in the market today as investors reacted
to the new economic policies. The kingdom's economy is growing.
"""

summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
print(summary[0]["summary_text"])
```

**Salida esperada:**
```
The king and queen rule the kingdom with wisdom and justice. Technology stocks rose in the market today.
```

Estos modelos requieren GPU para fine-tuning y generación, y pueden _alucinar_ (inventar hechos). Siempre revisa el resumen manualmente. Es [[Transfer Learning]] aplicado a generación: el modelo pre-entrenado se fine-tunea en pares (documento, resumen).

## 8. Common Mistakes

| Error | Consecuencia | Solución |
|---|---|---|
| Elegir mal número de tópicos | Tópicos mezclados o redundantes | Coherence score + inspección pyLDAvis |
| No revisar tópicos manualmente | Temas sin coherencia semántica pasan desapercibidos | Validación humana siempre |
| Alucinaciones en abstractive | Resumen inventa información inexistente | Threshold de probabilidad, beam search |
| Ignorar outliers en BERTopic | Documentos sin asignar a ningún tópico | Ajustar min_cluster_size en HDBSCAN |
| No preprocesar el texto | URLs, puntuación, stopwords contaminan tópicos | Pipeline de limpieza primero |

## Resumen

1. LDA modela documentos como mezcla de tópicos usando inferencia bayesiana; cada tópico es una distribución sobre palabras.
2. NMF factoriza la matriz TF-IDF en dos componentes no negativas, más rápido que LDA aunque sin fundamento probabilístico.
3. BERTopic usa embeddings contextuales + HDBSCAN para tópicos sin necesidad de fijar K.
4. TextRank es extractive: selecciona oraciones por centralidad en un grafo de similitud.
5. BART, T5 y PEGASUS son abstractive: generan resúmenes nuevos (riesgo de alucinaciones).
6. Coherence score (C_v) y pyLDAvis ayudan a evaluar calidad de tópicos; revisión humana es indispensable.

## Check Your Understanding

1. ¿Qué asume LDA sobre cómo se genera un documento? <!-- Que cada documento es una mezcla de temas y cada palabra se genera eligiendo primero un tema y luego una palabra de ese tema. -->
2. ¿Cuándo preferirías NMF sobre LDA? <!-- Cuando la escalabilidad y velocidad importan más que la interpretabilidad probabilística (ej. millones de documentos). -->
3. ¿Por qué BERTopic no necesita especificar el número de tópicos? <!-- Porque usa HDBSCAN, un algoritmo de clustering que determina el número de clusters automáticamente basado en la densidad. -->
4. ¿Qué diferencia hay entre summarization extractive y abstractive? <!-- Extractive selecciona oraciones originales; abstractive genera texto nuevo (puede alucinar). -->
5. ¿Qué mide el coherence score C_v? <!-- Si las palabras de un tópico aparecen juntas frecuentemente en el corpus real, indicando coherencia semántica. -->

## Where to Go Next

- [[Unsupervised Learning]]
- [[Text Preprocessing & Representation]]
- [[Word Embeddings (Word2Vec)]]
- [[Transformer Architecture]]
- [[Feature Engineering]]
- [[Probability]]
- [[Linear Algebra]]
