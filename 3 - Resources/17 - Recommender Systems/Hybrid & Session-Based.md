---
tags:
  - recommender-systems
  - hybrid
  - session-based
  - sequence-models
status: seedling
created: 2026-06-28
---

# Hybrid & Session-Based

## 1. Escenario de aprendizaje

Un usuario entra a tu tienda online. No ha iniciado sesión (no hay historial). Agrega una cámara réflex al carrito. En la página principal, el sistema debe recomendar productos. Collaborative Filtering (CF) no funciona — no hay usuario en la base de datos. Content-Based (CB) puede recomendar otras cámaras, pero el usuario ya tiene una en el carrito; probablemente necesita accesorios (trípode, lentes, memoria SD).

Además, la sesión es dinámica. Cada producto que agrega al carrito cambia el contexto. Lo que era relevante hace dos clics puede no serlo ahora. Las recomendaciones deben actualizarse en tiempo real.

Este escenario combina dos problemas que ningún enfoque puro resuelve bien: cold start (usuario no identificado) y dependencia temporal (sesión de compra). La solución: sistemas híbridos que combinan CF y CB, y modelos session-based que capturan la secuencia de interacciones.

## 2. Requisitos

- Python 3.9+
- scikit-learn
- Familiaridad con Collaborative Filtering y Content-Based Filtering
- Deseable: experiencia con redes neuronales (PyTorch/Keras) para session-based

## 3. Weighted Hybrid

Combina los scores de CF y CB usando un promedio ponderado.

```python
def weighted_hybrid_score(
    cf_scores: dict[int, float],
    cb_scores: dict[int, float],
    alpha: float = 0.5
) -> dict[int, float]:
    """
    Combina scores de CF y CB.
    alpha = peso de CF; (1-alpha) = peso de CB.

    alpha > 0.5: prioriza CF (mejor con datos suficientes)
    alpha < 0.5: prioriza CB (mejor en cold start)
    """
    all_items = set(cf_scores.keys()) | set(cb_scores.keys())
    hybrid = {}
    for item in all_items:
        cf = cf_scores.get(item, 0.0)
        cb = cb_scores.get(item, 0.0)
        hybrid[item] = alpha * cf + (1 - alpha) * cb
    return hybrid
```

```python
# Salida esperada:
# weighted_hybrid_score({1: 0.9, 2: 0.3}, {2: 0.8, 3: 0.7}, alpha=0.6)
# Retorna: {1: 0.54, 2: 0.50, 3: 0.28}
```

**Ajuste de alpha:** No hay un alpha universal. Debes calibrarlo con validación cruzada. Una estrategia: aprender alpha como un hiperparámetro más.

## 4. Feature-Augmented Hybrid

Usa los embeddings generados por CF (factorización de matrices o neural CF) como features adicionales para el modelo CB.

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Obtener embeddings latentes de CF
user_item_matrix = ...  # matriz usuarios x items
svd = TruncatedSVD(n_components=50)
item_factors = svd.components_.T  # (n_items, 50)

# 2. Concatenar con features de contenido
tfidf = TfidfVectorizer()
content_features = tfidf.fit_transform(items['description'])
# (n_items, n_terms)

combined_features = np.hstack([
    content_features.toarray(),
    item_factors
])
# (n_items, n_terms + 50)

# 3. Entrenar modelo CB sobre features combinados
model = LogisticRegression()
model.fit(combined_features, targets)
```

```python
# Salida esperada:
# Feature-augmented combina lo mejor de ambos: captura
# patrones latentes de CF (gustos implícitos de la comunidad)
# + explicabilidad de CB. El modelo resultante suele superar
# a ambos enfoques por separado.
```

## 5. Switching Hybrid

Usa CB para cold start y cambia a CF cuando hay suficientes datos del usuario.

```python
class SwitchingHybrid:
    def __init__(
        self,
        cb_recommender,
        cf_recomender,
        min_interactions: int = 5
    ):
        self.cb = cb_recommender
        self.cf = cf_recomender
        self.min_interactions = min_interactions

    def recommend(
        self,
        user_id: str,
        user_history: list[int],
        top_n: int = 10
    ) -> list[tuple[int, float]]:
        if len(user_history) < self.min_interactions:
            # Cold start: usar Content-Based
            return self.cb.recommend(user_history, top_n)
        else:
            # Datos suficientes: usar Collaborative Filtering
            return self.cf.recommend(user_id, top_n)
```

```python
# Salida esperada:
# Usuario nuevo (< 5 interacciones): usa CB.
# Usuario con historial: usa CF para mejor personalización.
# La transición es abrupta. Para suavizarla, puedes
# interpolar gradualmente el peso de CF vs CB a medida
# que crece el historial.
```

## 6. Session-Based

Las sesiones son secuencias de interacciones: `[item_3 → item_7 → item_2]`. El orden importa. Los modelos session-based capturan patrones secuenciales.

**Item2Vec:** Trata cada sesión como una "oración" de items. Usa Word2Vec para aprender embeddings de items basados en co-ocurrencia en sesiones.

```python
from gensim.models import Word2Vec

# Preparar sesiones como listas de items
sessions = [
    ['camera', 'tripod', 'sd_card'],
    ['camera', 'lens', 'camera_bag'],
    ['lens', 'filter', 'cleaning_kit'],
]

model = Word2Vec(
    sentences=sessions,
    vector_size=64,
    window=3,
    min_count=1,
    sg=1  # Skip-gram
)

# Item más similar a "camera"
model.wv.most_similar('camera', topn=3)
```

**GRU4Rec:** Red recurrente (GRU) que predice el siguiente item en la sesión.

```python
# Pseudocódigo: arquitectura GRU4Rec
# Input: secuencia de item_ids [3, 7, 2] -> embedding lookup
# Hidden: GRU(embedding_dim, hidden_dim)
# Output: softmax sobre todos los items -> probabilidad del próximo item
# Loss: cross-entropy entre item real y predicho
# Entrenamiento: sesiones de usuarios, prediciendo cada paso
```

**Transformers para sesiones:** Modelos como SASRec (Self-Attentive Sequential Recommendation) usan atención para capturar dependencias a largo plazo.

```python
# Pseudocódigo: SASRec
# Input: embeddings posicionales + item embeddings
# Transformer block: multi-head self-attention + FFN
# Output: score para cada item (qué tan probable es que sea el próximo)
# Ventaja: captura dependencias largas que GRU pierde
```

```python
# Salida esperada:
# Item2Vec: ('tripod', 0.85), ('lens', 0.72), ('sd_card', 0.68)
# GRU4Rec y SASRec: predicen el siguiente item dado el contexto
# de la sesión. SASRec suele superar a GRU4Rec cuando las
# sesiones tienen >10 items.
```

**Casos de uso por enfoque:**

| Enfoque | Cuándo funciona mejor | Ejemplo |
|---|---|---|
| Item2Vec | Sesiones cortas (<10), sin orden crítico | "Los que vieron X también vieron Y" |
| GRU4Rec | Sesiones medianas, orden importa | Recomendar siguiente producto en carrito |
| SASRec/Transformers | Sesiones largas, dependencias complejas | Secuencias de visualización en Netflix |

## 7. Evaluación de híbridos

Un sistema híbrido debe compararse contra cada uno de sus componentes individuales.

```python
from sklearn.metrics import ndcg_score

def evaluate_hybrid(
    recommender,
    test_sessions: list,
    k: int = 10
) -> float:
    """
    Evalúa un recomendador (híbrido o puro) con NDCG@k.
    """
    ndcg_scores = []
    for session in test_sessions:
        # Último item es el target; primeros N-1 son historial
        history = session[:-1]
        target = session[-1]

        recommendations = recommender.recommend(history, top_n=k)
        recommended_ids = [item for item, _ in recommendations]

        # NDCG: 1.0 si target está en top-1, 0 si no está en top-k
        relevance = [1 if item == target else 0 for item in recommended_ids]
        y_true = [relevance]
        y_score = [[score for _, score in recommendations]]
        ndcg_scores.append(ndcg_score(y_true, y_score, k=k))

    return np.mean(ndcg_scores)

# Comparar
print(f"CF-only NDCG@10: {evaluate_hybrid(cf_model, test_sessions)}")
print(f"CB-only NDCG@10: {evaluate_hybrid(cb_model, test_sessions)}")
print(f"Hybrid NDCG@10:  {evaluate_hybrid(hybrid_model, test_sessions)}")
```

```python
# Salida esperada:
# CF-only NDCG@10: 0.42
# CB-only NDCG@10: 0.38
# Hybrid NDCG@10:  0.51
# El híbrido debe superar a ambos componentes individuales.
# Si no es así, revisa los pesos o la estrategia de combinación.
```

## 8. Common Mistakes

**No calibrar los pesos del híbrido:** Usar alpha = 0.5 por defecto sin validar. Los pesos óptimos dependen del dataset y del stage del usuario. Realiza grid search sobre alpha.

**Session-based sin considerar orden temporal:** Item2Vec trata la sesión como una bolsa de items (bag-of-items) si no usas la ventana deslizante correctamente. El orden es la señal más importante en sesiones.

**Ignorar el decaimiento temporal en sesiones:** Un clic de hace 10 minutos debería pesar menos que el clic actual. Usa decay exponencial o positional encoding.

**Mezclar escalas de scores:** Si CF produce scores [0, 1] y CB produce scores [0, 100], combinarlos sin normalizar sesga el híbrido. Normaliza ambos al mismo rango.

**Evaluar el híbrido solo offline:** Un híbrido que funciona bien en offline puede fallar en online por latencia adicional (llamar a dos sistemas en lugar de uno). Mide latencia en producción.

```python
# Normalizar scores antes de combinar
def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    values = list(scores.values())
    if not values:
        return scores
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return {k: 0.5 for k in scores}
    return {
        k: (v - min_val) / (max_val - min_val)
        for k, v in scores.items()
    }
```

```python
# Salida esperada (Common Mistakes):
# Alpha sin calibrar: rendimiento subóptimo.
# Ignorar orden en sesiones: predictivo pobre.
# Scores sin normalizar: un componente domina.
# Sin medición de latencia: degradación en producción.
```

## Resumen

Los sistemas híbridos combinan CF y CB para superar las limitaciones individuales. Las estrategias principales son: weighted hybrid (promedio ponderado de scores), feature-augmented (embeddings de CF como features para CB), y switching (CB para cold start, CF para usuarios con historial). Session-based recommendation va un paso más allá, modelando secuencias de interacciones con Item2Vec, GRU4Rec o Transformers (SASRec). La evaluación debe comparar el híbrido contra cada componente individual, tanto en offline (NDCG, Recall) como en online (latencia, conversión).

## Check Your Understanding

1. ¿Cuándo preferirías un switching hybrid sobre un weighted hybrid?
<!-- Cuando hay una clara dicotomía entre cold start y usuarios con historial. Switching es más simple y evita el ruido de scores mal calibrados. Pero la transición es abrupta. -->

2. ¿Qué ventaja tiene feature-augmented sobre weighted hybrid?
<!-- No requiere combinar scores manualmente ni calibrar pesos. El modelo de CB aprende por sí mismo qué features de CF son útiles. Además puede capturar interacciones no lineales entre features de contenido y factores latentes. -->

3. En session-based recommendation, ¿por qué el orden de los items importa?
<!-- Porque la intención del usuario cambia durante la sesión. Alguien que agrega una cámara al carrito probablemente buscará accesorios después, no otra cámara. El orden revela la progresión de la intención de compra. -->

4. Tu hybrid weighted obtiene NDCG inferior al componente CF puro. ¿Qué posibles causas hay?
<!-- Peso de CB demasiado alto, scores no normalizados (CB domina con valores grandes), datos de contenido de baja calidad, o alpha mal calibrado. -->

5. ¿Por qué SASRec (Transformer) puede superar a GRU4Rec en sesiones largas?
<!-- Porque la atención permite capturar dependencias entre items distantes en la secuencia sin el cuello de botella del estado oculto recurrente. GRU4Rec comprime toda la historia en un vector oculto que puede saturarse en secuencias largas. -->

## Where to Go Next

Domina primero [[Collaborative Filtering]] y [[Content-Based Filtering]] — son los fundamentos que todo híbrido combina. [[Production Recommenders]] cubre cómo llevar estos modelos a producción con escalabilidad. [[RNNs & Sequence Models]] profundiza en GRU4Rec y arquitecturas recurrentes para sesiones. [[Transformers]] explica la arquitectura de atención usada en SASRec. [[Model Evaluation]] muestra cómo diseñar experimentos offline y online robustos. Y [[Feature Engineering]] te ayudará a crear mejores representaciones de contenido y sesiones.
