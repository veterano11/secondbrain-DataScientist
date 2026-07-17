---
tags:
  - recommender-systems
  - content-based-filtering
  - machine-learning
  - cold-start
status: seedling
created: 2026-06-28
---

# Content-Based Filtering

## 1. Escenario de aprendizaje

Un usuario nuevo se registra en tu plataforma de artículos científicos. No hay historial de clics, no hay compras previas, no hay ratings. Los sistemas de collaborative filtering no pueden ayudar porque no hay usuarios similares a los que referenciar — el problema clásico de cold start.

Pero el usuario indicó en el registro que le interesa "machine learning aplicado a salud". Tienes el texto completo de 50,000 artículos. Cada artículo tiene título, abstract, autores, categorías y palabras clave. No necesitas saber lo que otros usuarios hicieron — puedes recomendar basándote en el contenido de los artículos y los intereses explícitos del usuario.

Content-Based Filtering (CBF) resuelve cold start para items y usuarios nuevos, siempre que tengas metadata descriptiva. Este note cubre cómo representar items, construir perfiles de usuario, calcular similitud e implementar un sistema CBF completo.

## 2. Requisitos

- pandas
- scikit-learn (TfidfVectorizer, cosine_similarity)
- numpy
- Familiaridad con algebraic lineal básica (vectores, matrices)

## 3. Representación de items

Para comparar items, necesitas una representación vectorial. La elección depende del tipo de datos.

**Texto (TF-IDF):** Convierte documentos en vectores donde cada dimensión es un término y el valor es la importancia del término en el documento.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning for healthcare diagnostics",
    "deep reinforcement learning for robotics",
    "natural language processing for clinical notes"
]

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(documents)
print(tfidf_matrix.shape)  # (3 documentos, 1000 términos)
```

**Categorías (One-hot / Multi-hot):** Para variables categóricas como género, categoría, país.

```python
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

items = pd.DataFrame({
    'item_id': [1, 2, 3],
    'genres': [
        ['science_fiction', 'action'],
        ['drama', 'romance'],
        ['science_fiction', 'drama']
    ]
})

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(items['genres'])
# Cada fila es un vector binario de géneros
```

**Embeddings (numérico):** Si tienes imágenes o texto, puedes usar modelos pre-entrenados (ResNet, BERT) para extraer embeddings densos.

```python
# Pseudocódigo: usar embeddings de BERT
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(documents)
```

```python
# Salida esperada:
# TF-IDF: matriz dispersa de (n_docs x vocab_size).
# One-hot: matriz densa de (n_items x n_categories).
# Embeddings: matriz densa de (n_items x embedding_dim), típicamente 384 o 768.
```

## 4. Perfil de usuario

El perfil de usuario es un vector en el mismo espacio que los items. La estrategia más común: promedio ponderado de los vectores de items que el usuario ha consumido o valorado.

```python
import numpy as np

def build_user_profile(
    item_vectors: np.ndarray,
    item_ids: list[int],
    user_item_ids: list[int],
    ratings: list[float] = None
) -> np.ndarray:
    """
    Construye perfil de usuario como promedio ponderado
    de vectores de items que ha consumido.

    Args:
        item_vectors: matriz (n_items x n_features)
        item_ids: lista de IDs de items
        user_item_ids: IDs de items que el usuario consumió
        ratings: pesos opcionales (ej: rating explícito)
    """
    indices = [item_ids.index(iid) for iid in user_item_ids]
    user_vectors = item_vectors[indices]

    if ratings:
        weights = np.array(ratings).reshape(-1, 1)
        profile = np.average(user_vectors, axis=0, weights=weights.flatten())
    else:
        profile = user_vectors.mean(axis=0)

    return profile

# Ejemplo: usuario que vio 3 artículos y le gustaron
profile = build_user_profile(
    item_vectors=tfidf_matrix.toarray(),
    item_ids=[1, 2, 3],
    user_item_ids=[1, 3],
    ratings=[5, 4]
)
```

```python
# Salida esperada:
# profile es un vector de la misma dimensionalidad que
# item_vectors. Representa el "item ideal" del usuario
# en el espacio de features.
```

## 5. Similitud

Para recomendar, calculamos la similitud entre el perfil del usuario y todos los items.

```python
from sklearn.metrics.pairwise import cosine_similarity

def recommend_items(
    user_profile: np.ndarray,
    item_vectors: np.ndarray,
    item_ids: list[int],
    top_n: int = 10
) -> list[tuple[int, float]]:
    """
    Calcula similitud del perfil contra todos los items
    y retorna los top-N.
    """
    similarities = cosine_similarity(
        user_profile.reshape(1, -1),
        item_vectors
    ).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [(item_ids[i], similarities[i]) for i in top_indices]

recommendations = recommend_items(profile, tfidf_matrix.toarray(), [1, 2, 3])
```

**¿Cuándo usar cada métrica?**

| Métrica | Cuándo usarla | Ejemplo |
|---|---|---|
| Cosine similarity | Datos dispersos (TF-IDF), magnitudes no importan | Texto, ratings implícitos |
| Euclidean distance | Datos densos, magnitudes importan | Embeddings, features numéricas |
| Jaccard similarity | Sets binarios, categorías | One-hot genres, tags |

```python
# Salida esperada:
# recommend_items retorna [(item_id, score), ...]
# score = cosine similarity entre 0 (ortogonal) y 1 (idéntico).
# Los items ya consumidos deben excluirse de las recomendaciones.
```

## 6. Implementación completa

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features
        )
        self.item_vectors = None
        self.item_ids = None

    def fit(self, items: pd.DataFrame, text_column: str = 'description'):
        """Construye la matriz item-features desde los datos."""
        self.item_ids = items['item_id'].tolist()
        self.item_vectors = self.vectorizer.fit_transform(
            items[text_column]
        )

    def recommend(
        self,
        user_item_ids: list[int],
        top_n: int = 10,
        exclude_consumed: bool = True
    ) -> list[tuple[int, float]]:
        """Recomienda items basado en el historial del usuario."""
        indices = [self.item_ids.index(iid) for iid in user_item_ids]
        user_vector = self.item_vectors[indices].mean(axis=0)

        similarities = cosine_similarity(
            user_vector,
            self.item_vectors
        ).flatten()

        if exclude_consumed:
            consumed_indices = set(indices)
            mask = np.ones(len(similarities), dtype=bool)
            mask[list(consumed_indices)] = False
            similarities[~mask] = -1

        top_indices = np.argsort(similarities)[::-1][:top_n]
        return [(self.item_ids[i], similarities[i])
                for i in top_indices if similarities[i] > 0]

# Uso
recsys = ContentBasedRecommender()
recsys.fit(articles_df, text_column='abstract')
recommendations = recsys.recommend(
    user_item_ids=[101, 203, 305],
    top_n=5
)
```

```python
# Salida esperada:
# recommendations = [(42, 0.87), (15, 0.82), (73, 0.79), ...]
# Item 42 tiene el abstract más similar al perfil del usuario.
```

## 7. Ventajas y limitaciones

**Ventajas:**
- Resuelve cold start para nuevos items (basta con su metadata)
- No requiere datos de otros usuarios — funciona con un solo usuario
- Las recomendaciones son explicables: "porque este artículo tiene palabras similares a los que leíste"
- No sufre de popularity bias (no depende de ratings agregados)

**Limitaciones:**
- No hay serendipia: solo recomienda contenido similar a lo que ya consumió
- Requiere metadata rica y relevante. Si los abstracts son pobres, las recomendaciones serán pobres
- No captura preferencias latentes (ej: "me gusta ciencia ficción pero no distopías")
- Sobreespecialización: el usuario se queda en una burbuja de contenido similar

```python
# Salida esperada:
# Un usuario que solo consumió artículos de "reinforcement
# learning" recibirá solo artículos similares. Nunca
# descubrirá "natural language processing" aunque pueda
# interesarle. Esto es el "filter bubble" de CBF.
```

## 8. Common Mistakes

**No normalizar features numéricas:** Si las features tienen escalas distintas (edad 0-100 vs ingresos 0-1M), la similitud estará dominada por las de mayor magnitud.

```python
from sklearn.preprocessing import StandardScaler

# Siempre normalizar features numéricas
scaler = StandardScaler()
numeric_features = scaler.fit_transform(items[['age', 'income', 'tenure']])
```

**TF-IDF sin preprocesamiento:** Aplicar TF-IDF directamente sobre texto sin limpiar (stop words, stemming, lowercase) produce vectores ruidosos con términos irrelevantes.

**Similitud sobre categorías con one-hot denso:** Si tienes 500 categorías representadas como one-hot de 500 dimensiones, la similitud coseno entre dos vectores será casi siempre 0. Considera embeddings o reducir dimensionalidad.

**No excluir items ya consumidos:** Recomendar items que el usuario ya vio/consumió es frustrante. Siempre filtra el historial.

**Ignorar el balance entre features:** Si un item tiene 100 palabras de descripción y 3 categorías, TF-IDF dominará sobre one-hot. Normaliza o pondera los bloques de features.

```python
# Salida esperada (Common Mistakes):
# Features sin normalizar: similitud dominada por income.
# TF-IDF sin stop_words: términos como "the", "is" contaminan.
# One-hot denso: similitudes cercanas a cero.
# Items ya consumidos en recomendaciones: user experience pobre.
```

## Resumen

Content-Based Filtering recomienda items basándose en la similitud entre el perfil del usuario y la representación de los items. La pipeline esencial es: representar items como vectores (TF-IDF para texto, one-hot para categorías, embeddings para datos complejos), construir el perfil del usuario como agregación de items consumidos, calcular similitud (cosine similarity es la opción más común), y generar el top-N excluyendo items ya vistos. CBF es ideal para cold start y ofrece recomendaciones explicables, pero sufre de sobreespecialización y requiere metadata de calidad.

## Check Your Understanding

1. ¿Por qué TF-IDF es preferible a Bag-of-Words para representar texto en CBF?
<!-- TF-IDF pondera términos raros (informativos) más alto que términos frecuentes (como "the", "is"). Esto captura mejor la relevancia temática. -->

2. Un usuario nuevo indica que le gusta "horror y comedia". Dos items tienen exactamente los mismos vectores TF-IDF. ¿Por qué podrían ser recomendaciones muy diferentes?
<!-- TF-IDF captura el texto, no las categorías. Si los items tienen textos similares pero géneros diferentes, CBF con solo TF-IDF no los distinguirá. Necesitas incluir features categóricas. -->

3. ¿Qué estrategia usarías para mitigar la sobreespecialización en CBF?
<!-- Añadir diversidad explícita en el ranking, introducir exploración aleatoria, usar un enfoque híbrido con collaborative filtering, o agregar un término de "novedad" en el score. -->

4. Tu jefe pregunta: "¿Por qué el recomendador solo sugiere artículos de deep learning si el usuario también podría estar interesado en estadística?". ¿Cómo respondes?
<!-- Porque CBF recomienda basado en similitud con el historial. Para romper la burbuja, necesitamos incorporar diversidad o usar un enfoque híbrido. Content-Based es bueno para precisión, no para descubrimiento. -->

5. ¿Cuándo usarías Jaccard similarity en lugar de cosine similarity?
<!-- Cuando las features son binarias y el tamaño del set importa (ej: overlap de tags entre dos películas). Jaccard = |A ∩ B| / |A ∪ B|. Cosine similarity considera la magnitud del vector, Jaccard solo la intersección relativa. -->

## Where to Go Next

[[Recommendation Fundamentals]] introduce los conceptos base de sistemas de recomendación que todo profesional debe conocer. [[Collaborative Filtering]] es el complemento natural a CBF — aprende cómo usar el comportamiento de otros usuarios para recomendar. [[Hybrid & Session-Based]] muestra cómo combinar CBF y CF para obtener lo mejor de ambos mundos. [[Feature Engineering]] profundiza en cómo crear representaciones de items más informativas. [[Text Preprocessing & Representation]] cubre las técnicas de NLP necesarias para mejorar TF-IDF. [[Unsupervised Learning]] explora cómo clustering y reducción de dimensionalidad pueden mejorar CBF. Y [[Linear Algebra]] te dará la base matemática para entender vectores, matrices y similitudes a fondo.
