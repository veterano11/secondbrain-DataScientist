---
tags:
  - recommender-systems
  - production
  - cold-start
  - exploration-exploitation
status: seedling
created: 2026-06-28
---

# Production Recommenders

## 1. Escenario de aprendizaje

Tu modelo de recomendación alcanzó NDCG=0.85 en validación offline. El equipo celebra. Pero al desplegarlo en producción con 1M de usuarios activos diarios, los números reales no acompañan: los usuarios hacen clic pero no convierten. Sesiones de compra que antes cerraban ahora se abandonan. Los usuarios nuevos reciben recomendaciones genéricas de los items más populares. Y los items nuevos — lanzados esta semana — nunca aparecen en las recomendaciones.

Los problemas reales en producción son distintos de los que ves en offline. Cold start, popularity bias, exploración vs explotación, re-ranking por reglas de negocio, y evaluación online son los desafíos que separan un modelo que funciona en un notebook de un sistema que funciona para millones de usuarios.

## 2. Requisitos

- Experiencia con sistemas de recomendación (CF, CB, híbridos)
- Familiaridad con conceptos de producción: latencia, escalabilidad, A/B testing
- Deseable: experiencia con bandit algorithms y MAB

## 3. Cold start

El cold start tiene dos caras: usuarios nuevos e items nuevos.

**Usuario nuevo:** No tienes historial de interacciones. Estrategias:

```python
# Estrategia 1: Encuesta de onboarding
onboarding_preferences = {
    'user_123': {
        'categories': ['electronics', 'photography'],
        'price_range': 'mid_range',
        'brand_affinities': ['sony', 'canon']
    }
}

def recommend_new_user(preferences: dict, top_n: int = 10) -> list:
    """Usa preferencias explícitas para recomendación inicial."""
    items = filter_by_category(preferences['categories'])
    items = filter_by_price(items, preferences['price_range'])
    items = boost_by_brand(items, preferences['brand_affinities'])
    return rank_by_popularity(items)[:top_n]

# Estrategia 2: Popularidad + segmentación demográfica
def recommend_by_demographics(
    age_group: str,
    country: str,
    top_n: int = 10
) -> list:
    """Recomienda lo más popular en el segmento del usuario."""
    segment_items = get_top_items_for_segment(age_group, country)
    return segment_items[:top_n]
```

**Item nuevo:** No tiene interacciones, no aparece en CF. Solución: content-based cold start.

```python
def score_new_item(
    new_item_features: dict,
    existing_items: pd.DataFrame,
    user_profiles: dict
) -> float:
    """Estima el score de un item nuevo usando similitud de contenido."""
    # Encontrar items similares por metadata
    similar_items = find_similar_by_content(new_item_features, existing_items)
    # Usar scores agregados de items similares como proxy
    estimated_score = aggregate_user_scores(similar_items, user_profiles)
    return estimated_score
```

```python
# Salida esperada:
# Usuario nuevo sin onboarding: recibe top 10 populares del día.
# Usuario nuevo con onboarding: recibe items filtrados por preferencias.
# Item nuevo: recibe score estimado basado en items similares.
# Ambas estrategias se actualizan cuando el usuario/item acumula
# interacciones reales.
```

## 4. Popularity bias

Los items populares se recomiendan más → obtienen más clics → se vuelven más populares. Es un feedback loop que ahoga items de nicho y nuevos.

```python
import numpy as np

def mitigate_popularity_bias(
    scores: np.ndarray,
    item_popularity: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Penaliza scores de items muy populares.

    Args:
        scores: scores de recomendación (ej: predicted rating)
        item_popularity: frecuencia normalizada [0, 1]
        alpha: fuerza de la penalización (0 = sin penalización)
    """
    # Penalización: items más populares pierden más score
    penalty = 1.0 - alpha * item_popularity
    return scores * penalty
```

```python
# Salida esperada:
# Sin penalización: los mismos 5 items populares aparecen
# en todas las recomendaciones.
# Con alpha=0.3: items de nicho comienzan a aparecer en
# posiciones recomendables, manteniendo calidad.
```

**Técnicas adicionales:**
- **Inverse popularity weighting:** Divide el score por la popularidad
- **Re-ranking por diversidad:** MMR (Maximal Marginal Relevance) asegura variedad
- **Exposición justa:** Fairness-aware ranking limita la exposición de items populares

## 5. Exploración vs Explotación

El dilema fundamental: ¿recomiendas lo que sabes que funciona (explotación) o pruebas cosas nuevas (exploración)?

```python
import random
import numpy as np

# Epsilon-Greedy
class EpsilonGreedy:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def select_item(self, item_scores: dict[int, float]) -> int:
        if random.random() < self.epsilon:
            return random.choice(list(item_scores.keys()))
        return max(item_scores, key=item_scores.get)

# Upper Confidence Bound (UCB)
class UCB:
    def __init__(self, n_items: int):
        self.counts = np.zeros(n_items)
        self.values = np.zeros(n_items)

    def select_item(self, t: int) -> int:
        ucb_scores = self.values + np.sqrt(
            2 * np.log(t + 1) / (self.counts + 1e-6)
        )
        return np.argmax(ucb_scores)

    def update(self, item_id: int, reward: float):
        self.counts[item_id] += 1
        n = self.counts[item_id]
        value = self.values[item_id]
        self.values[item_id] = ((n - 1) / n) * value + (1 / n) * reward

# Thompson Sampling
class ThompsonSampling:
    def __init__(self, n_items: int):
        self.alpha = np.ones(n_items)  # éxitos
        self.beta = np.ones(n_items)   # fracasos

    def select_item(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, item_id: int, reward: bool):
        if reward:
            self.alpha[item_id] += 1
        else:
            self.beta[item_id] += 1
```

```python
# Salida esperada:
# Epsilon-Greedy: 10% de tráfico exploratorio (aleatorio).
# UCB: explora items inciertos (pocas observaciones) más
# que items con alta varianza.
# Thompson Sampling: muestrea de distribuciones beta —
# naturalmente balancea exploración y explotación.
```

**MAB Contextual (LinUCB):** Usa features del usuario/item para decidir exploración.

```python
# LinUCB: UCB con regresión lineal por item.
# Selecciona el item con mayor upper confidence bound
# donde el bound depende del contexto (features del usuario).
```

## 6. Re-ranking

Antes de mostrar recomendaciones al usuario, aplicas reglas de negocio y diversidad.

```python
def rerank_candidates(
    candidates: list[tuple[int, float]],
    purchased_items: set[int],
    diversity_weight: float = 0.3
) -> list[tuple[int, float]]:
    """
    Re-ranking con reglas de negocio y diversidad.
    """
    # Regla 1: No recomendar items comprados
    candidates = [
        (item, score) for item, score in candidates
        if item not in purchased_items
    ]

    # Regla 2: Frescura — items de la última semana tienen boost
    boosted = []
    for item, score in candidates:
        if is_new_item(item, days=7):
            score *= 1.2  # boost 20%
        boosted.append((item, score))

    # Regla 3: MMR para diversidad
    return mmr_rerank(boosted, diversity_weight)

def mmr_rerank(
    candidates: list[tuple[int, float]],
    lambda_param: float = 0.5
) -> list[tuple[int, float]]:
    """
    Maximal Marginal Relevance: balancea relevancia y diversidad.
    """
    selected = []
    remaining = list(candidates)

    for _ in range(min(10, len(candidates))):
        mmr_scores = []
        for item, score in remaining:
            similarity_to_selected = max(
                [compute_similarity(item, s_item)
                 for s_item, _ in selected],
                default=0.0
            )
            mmr = (1 - lambda_param) * score - lambda_param * similarity_to_selected
            mmr_scores.append(mmr)

        best_idx = np.argmax(mmr_scores)
        selected.append(remaining.pop(best_idx))

    return selected
```

```python
# Salida esperada:
# Antes de re-ranking: recomendaciones relevantes pero repetitivas.
# Después: variedad de categorías, sin items comprados,
# con items nuevos destacados (boost).
```

## 7. Online evaluation

Offline no es suficiente. Necesitas medir el impacto real en el negocio.

**A/B Testing:** Divide usuarios en grupos, cada grupo recibe un recomendador diferente.

```python
# Pseudocódigo para A/B test
control_model = BaselineRecommender()
treatment_model = NewHybridRecommender()

def get_recommendations(user_id: str):
    if user_id in control_group:
        return control_model.recommend(user_id)
    else:
        return treatment_model.recommend(user_id)

# Métricas: CTR, conversión, revenue por usuario, diversidad
```

**Interleaving Experiments:** En lugar de mostrar solo un modelo, intercala recomendaciones de ambos en la misma lista y observa cuál recibe más clics.

```python
def interleaved_ranking(
    model_a_items: list[int],
    model_b_items: list[int]
) -> list[int]:
    """Intercala resultados de dos modelos para comparación directa."""
    result = []
    i, j = 0, 0
    turn = 0
    while i < len(model_a_items) and j < len(model_b_items):
        if turn % 2 == 0:
            result.append(model_a_items[i])
            i += 1
        else:
            result.append(model_b_items[j])
            j += 1
        turn += 1
    result.extend(model_a_items[i:])
    result.extend(model_b_items[j:])
    return result
```

```python
# Salida esperada:
# A/B test: 50% usuarios ven control, 50% ven treatment.
# Interleaving: todos los usuarios ven mezcla de ambos.
# Interleaving tiene mayor poder estadístico (cada usuario
# es su propio control).
```

## 8. Common Mistakes

**No medir diversidad:** Un modelo puede tener alto CTR pero mostrar siempre los mismos items. Mide el número de items únicos recomendados por usuario y la cobertura del catálogo.

**Popularidad canibalizando personalización:** Cuando el 80% de las recomendaciones son los mismos 100 items populares, no hay personalización real. Monitorea qué fracción de recomendaciones son personalizadas vs populares.

**Cold start sin plan de transición:** Una vez que el usuario nuevo acumula interacciones, el sistema debe migrar de recomendaciones populares a personalizadas. Sin esta transición, el usuario nunca recibe personalización real.

**Exploración sin métricas de follow-up:** Si exploras y el usuario hace clic, ¿compra? Si no hay seguimiento de conversión, la exploración puede optimizar CTR pero no revenue.

**Evaluación offline como proxy único:** NDCG alto no garantiza éxito en producción. Los datos offline tienen sesgo de popularidad y no capturan el efecto de las recomendaciones en el comportamiento futuro.

```python
# Salida esperada (Common Mistakes):
# Popularity bias no medido: cobertura del catálogo baja.
# Transición cold start no implementada: usuarios nuevos
# nunca reciben recomendaciones personalizadas.
# Exploración no trackeada: clickbait que no convierte.
```

## Resumen

Los sistemas de recomendación en producción enfrentan desafíos que no aparecen en validación offline. El cold start se aborda con encuestas, popularidad segmentada y content-based para items nuevos. El popularity bias se mitiga con penalización por popularidad, re-ranking por diversidad (MMR) y fairness-aware ranking. La exploración vs explotación se maneja con epsilon-greedy, UCB y Thompson Sampling. El re-ranking aplica reglas de negocio (no recomendar comprados, boost a items nuevos, diversidad). Y la evaluación online requiere A/B testing e interleaving experiments con métricas de negocio reales (CTR, conversión, revenue).

## Check Your Understanding

1. ¿Por qué un modelo con NDCG alto en offline puede fallar en producción?
<!-- NDCG offline tiene sesgo de popularidad (los datos reflejan el comportamiento del sistema anterior), no captura el efecto de las recomendaciones en el comportamiento futuro, y no mide métricas de negocio como conversión o revenue. -->

2. ¿Cuál es la diferencia entre epsilon-greedy y Thompson Sampling para exploración?
<!-- Epsilon-greedy elige aleatoriamente con probabilidad fija epsilon. Thompson Sampling muestrea de distribuciones de probabilidad, explorando items inciertos pero evitando items que sabemos que son malos. Thompson Sampling es más eficiente porque la exploración se adapta a la incertidumbre. -->

3. ¿Cómo detectas popularity bias en tu sistema?
<!-- Mide la cobertura del catálogo (% de items únicos recomendados), la concentración de impresiones en los top-N items (curva de Lorenz), y compara la distribución de recomendaciones con la distribución ideal. -->

4. Tu equipo quiere lanzar un nuevo modelo. ¿Qué experimento diseñarías para evaluarlo en producción?
<!-- A/B test con usuarios divididos aleatoriamente (50/50 o 90/10). Métricas primarias: CTR y conversión. Secundarias: diversidad, cobertura, revenue por usuario. Duración: al menos 1-2 semanas para capturar efectos de semana. -->

5. Un usuario nuevo recibe recomendaciones populares durante 3 semanas sin transición a personalizadas. ¿Qué problema hay?
<!-- El usuario nunca experimenta personalización. El sistema no tiene un plan de transición de cold start a warm start basado en número de interacciones. Después de N interacciones (ej: 10 clics), el usuario debería recibir recomendaciones personalizadas. -->

## Where to Go Next

[[Recommendation Fundamentals]] proporciona la base teórica de todos los conceptos aquí mencionados. [[Collaborative Filtering]] profundiza en el componente principal de muchos sistemas de producción. [[Hybrid & Session-Based]] expande cómo combinar múltiples señales en producción. [[A-B Testing]] cubre diseño estadístico de experimentos online. [[Model-Based RL]] explora el framing de recomendación como problema de reinforcement learning (exploración, reward, policy). [[Model Monitoring]] es esencial para detectar degradación en producción. Y [[Feature Stores]] muestra cómo gestionar features en producción a escala.
