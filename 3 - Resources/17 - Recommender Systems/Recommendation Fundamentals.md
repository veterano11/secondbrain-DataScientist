---
tags: [recommender-systems, fundamentals, machine-learning, evaluation]
status: seedling
created: 2026-06-28
---

# Recommendation Fundamentals

## 1. Escenario de aprendizaje

Trabajas en el equipo de datos de un e-commerce que quiere lanzar un sistema de recomendaciones. El negocio es claro: "cuando un usuario ve un producto, muéstrale otros productos que le puedan interesar". Pero apenas empiezas a investigar, te das cuenta de que hay decenas de enfoques: filtrado colaborativo, contenido basado en contenido, factorización de matrices, deep learning, bandits, y eso es solo la punta.

Antes de elegir un algoritmo, necesitas entender los fundamentos: ¿qué tipos de feedback existen? ¿cómo mides si una recomendación es buena? ¿por qué el modelo más simple (recomendar lo más popular) suele ser sorprendentemente efectivo?

Esta nota cubre los fundamentos que todo ingeniero de recomendaciones debe saber: tipos de feedback, métricas offline y online, el problema de evaluación, y el baseline de popularidad. Es la base para entender [[Collaborative Filtering]], [[Content-Based Filtering]] y [[Hybrid & Session-Based]] systems.

## 2. Requisitos

- Python, pandas, numpy, scikit-learn instalados
- Conceptos básicos de [[Supervised Learning]] y [[Model Evaluation]]
- Un dataset de ejemplo (usaremos MovieLens pequeño: 100k ratings)

## 3. Tipos de Feedback

Todo sistema de recomendaciones funciona con feedback de usuarios. Hay dos grandes familias:

**Feedback explícito:** El usuario dice explícitamente qué le gusta. Ejemplos: estrellas (1-5), like/dislike, thumbs up/down. Es información de alta calidad, pero escasa: los usuarios no rating cada cosa que consumen.

**Feedback implícito:** El usuario no califica, pero su comportamiento revela preferencias. Ejemplos: clics, vistas, tiempo de permanencia, compras, scroll. Es abundante (cada interacción genera datos) pero ruidoso: un clic no significa que le guste, puede ser un error o curiosidad.

```python
import pandas as pd
import numpy as np

# Feedback explícito: matriz usuario-item con ratings
explicit = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3],
    "item_id": [101, 102, 101, 103, 102],
    "rating": [5, 3, 4, 2, 5]
})

# Feedback implícito: matriz usuario-item con conteo de clics
implicit = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 3],
    "item_id": [101, 102, 103, 101, 103, 102],
    "click": [1, 1, 1, 1, 0, 1]
})

print("Explícito (sparse):")
print(explicit)
print("\nImplícito (más denso):")
print(implicit)
```

Salida esperada:
```
Explícito (sparse):
   user_id  item_id  rating
0        1      101       5
1        1      102       3
2        2      101       4
3        2      103       2
4        3      102       5

Implícito (más denso):
   user_id  item_id  click
0        1      101      1
1        1      102      1
2        1      103      1
3        2      101      1
4        2      103      0
5        3      102      1
```

En la práctica, los sistemas exitosos combinan ambos: feedback implícito para abundancia y explícito para calibración. Amazon usa compras (implícito) y ratings (explícito). YouTube usa vistas, tiempo de reproducción, likes y shares.

## 4. Métricas Offline

Antes de lanzar un modelo a producción, necesitas medir su calidad en datos históricos. Las métricas offline más comunes son:

**Precision@k:** ¿Cuántos de los k items recomendados son relevantes?
```
precision@k = (items relevantes en top-k) / k
```

**Recall@k:** ¿Cuántos de los items relevantes totales fueron recomendados en top-k?
```
recall@k = (items relevantes en top-k) / (total items relevantes)
```

**NDCG (Normalized Discounted Cumulative Gain):** Mide el ranking: un item relevante en posición 1 vale más que en posición 5. Es la métrica estándar para sistemas de recomendación porque penaliza poner items relevantes al final de la lista.

**MAP (Mean Average Precision):** Promedio de precision@k para cada usuario. Buena cuando tienes múltiples niveles de relevancia.

**RMSE (Root Mean Square Error):** Para predicción de ratings exactos, no para ranking. Mide el error entre rating predicho y real.

```python
from sklearn.metrics import ndcg_score
import numpy as np

# Scores predichos por el modelo para 3 usuarios y 5 items
y_true = np.array([
    [3, 2, 0, 0, 1],  # Usuario 1: relevancia real
    [1, 0, 0, 4, 5],
    [0, 0, 2, 0, 0]
])

y_score = np.array([
    [2.5, 1.8, 0.2, 0.1, 0.9],  # Scores del modelo
    [1.2, 0.1, 0.3, 3.8, 4.9],
    [0.1, 0.0, 1.5, 0.2, 0.0]
])

ndcg = ndcg_score(y_true, y_score, k=3)
print(f"NDCG@3: {ndcg:.3f}")

def precision_at_k(y_true, y_score, k=3):
    top_k = np.argsort(y_score, axis=1)[:, -k:][:, ::-1]
    relevant = np.take_along_axis(y_true, top_k, axis=1)
    return np.mean(relevant > 0)

print(f"Precision@3: {precision_at_k(y_true, y_score, 3):.3f}")
```

Salida esperada:
```
NDCG@3: 0.827
Precision@3: 0.667
```

Ninguna métrica cuenta toda la historia. NDCG es buena para ranking, RMSE para predicción exacta, precision/recall para cobertura. Siempre reporta un conjunto de métricas, no una sola.

## 5. Métricas Online

Las métricas offline te dicen cómo se comporta el modelo en datos pasados, pero no capturan el comportamiento real de los usuarios. Las métricas online se miden en producción con usuarios reales:

- **CTR (Click-Through Rate):** Clics / Impresiones. Mide qué tan atractiva es la recomendación.
- **Conversion Rate:** Compras / Impresiones. Mide qué tan efectiva es para generar ingresos.
- **Engagement:** Tiempo en sesión, páginas visitadas, tasa de retención. Mide satisfacción a largo plazo.
- **Novelty:** ¿Qué tan diferentes son las recomendaciones de lo que el usuario ya ha visto?
- **Diversity:** ¿Qué tan variados son los items recomendados? (vs recomendar solo del mismo género)

```python
# Simulación de métricas online
impressions = 10000
clicks = 450
purchases = 32

ctr = clicks / impressions * 100
conversion_rate = purchases / impressions * 100

print(f"CTR: {ctr:.2f}%")
print(f"Conversion Rate: {conversion_rate:.2f}%")
```

Salida esperada:
```
CTR: 4.50%
Conversion Rate: 0.32%
```

Las métricas online son la verdad definitiva. Un modelo con NDCG perfecto pero CTR bajo es peor que un modelo con NDCG mediocre pero CTR alto. [[A-B Testing]] es la herramienta estándar para comparar modelos en producción sin arriesgar la experiencia de todos los usuarios.

## 6. El Problema de la Evaluación

Evaluar sistemas de recomendación es notoriamente difícil por varias razones:

**Offline vs Online gap:** Un modelo con excelentes métricas offline puede fracasar en producción. ¿Por qué? Porque los datos offline están sesgados por el sistema anterior: los usuarios solo interactuaron con items que el sistema anterior les mostró. Esto se llama **exposure bias** o **position bias**.

**Data bias:** Los datos de entrenamiento suelen venir de usuarios activos que dejan muchos ratings. Pero los usuarios que más ratings dejan no son representativos del total. Si entrenas solo con usuarios power-users, tu modelo será malo para usuarios nuevos o casuales.

**El problema del contrafactual:** No puedes saber qué hubiera pasado si mostraban otro item. Si un usuario no hizo clic en un item, ¿es porque no le gustaba o porque no lo vio? No lo sabes con certeza.

```python
# Simulación de exposure bias
# Los items en posición 1-3 reciben 80% de los clics
# No porque sean mejores, sino porque están más visibles
positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
exposure_bias = {p: 0.3 * (0.6 ** (p-1)) for p in positions}  # Decae exponencialmente

clicks_by_position = {p: int(1000 * exposure_bias[p]) for p in positions}
print("Clics por posición (sesgados por visibilidad):")
for p, c in clicks_by_position.items():
    print(f"  Posición {p}: {c} clics")
```

Salida esperada:
```
Clics por posición (sesgados por visibilidad):
  Posición 1: 300 clics
  Posición 2: 180 clics
  Posición 3: 108 clics
  Posición 4: 65 clics
  Posición 5: 39 clics
  Posición 6: 23 clics
  Posición 7: 14 clics
  Posición 8: 8 clics
  Posición 9: 5 clics
  Posición 10: 3 clics
```

Para mitigar estos problemas usa técnicas como: evaluación con datos de un sistema anterior (si tienes logs), métodos de debiasing (IPW, inverse propensity weighting), y siempre complementa offline metrics con un [[A-B Testing]] en producción.

## 7. Popularity Baseline — El Modelo Más Simple

Antes de implementar algoritmos sofisticados, implementa un baseline de popularidad. Es el modelo más simple posible: recomienda a todos los usuarios los items más populares globalmente.

Sorprendentemente, este baseline suele ser muy competitivo, especialmente en plataformas nuevas sin datos de usuarios. Netflix, Spotify y Amazon lo usan como baseline contra el cual comparar modelos nuevos.

```python
from collections import Counter

# Dataset: interacciones usuario-item (sin ratings, solo ocurrencias)
interactions = [
    ("user1", "item_A"), ("user1", "item_B"), ("user2", "item_A"),
    ("user2", "item_C"), ("user3", "item_A"), ("user3", "item_D"),
    ("user4", "item_B"), ("user4", "item_A"), ("user5", "item_E"),
]

item_counts = Counter(item for _, item in interactions)
popular_items = [item for item, _ in item_counts.most_common(5)]

def recommend_popular(user_id: str, k: int = 3) -> list:
    return popular_items[:k]

print("Recomendación para usuario nuevo:")
print(recommend_popular("new_user", k=3))
```

Salida esperada:
```
Recomendación para usuario nuevo:
['item_A', 'item_B', 'item_C']
```

El baseline de popularidad es útil porque:
1. Es rápido de implementar (una línea de Counter)
2. Es barato de computar (una consulta a Redis)
3. Te da una cota inferior: si tu modelo sofisticado no supera la popularidad, algo está mal
4. Es excelente para cold start (usuarios nuevos sin historial)

Pero tiene limitaciones: las recomendaciones son idénticas para todos los usuarios (no personalizan). Es el punto de partida, no el destino. [[Statistics]] básicas te ayudarán a entender por qué la popularidad es tan efectiva inicialmente.

## 8. Common Mistakes

- **Evaluar con datos sesgados:** Usar solo datos de usuarios activos para entrenar y evaluar. Los resultados no generalizan a usuarios casuales. Siempre segmenta tus métricas por tipo de usuario.
- **Ignorar popularity bias:** Los modelos de collaborative filtering tienden a recomendar items populares porque hay más datos sobre ellos. Esto crea un bucle: los items populares se recomiendan más, se vuelven aún más populares, y los items de nicho mueren.
- **Usar RMSE para ranking:** RMSE mide error de predicción exacta de ratings. Para recomendaciones no importa si predices 3.5 vs 4.2, importa que los items relevantes aparezcan arriba.
- **Confundir correlación con causalidad:** Que los usuarios compren items recomendados no significa que la recomendación causó la compra. Quizás ya iban a comprar ese producto.
- **No tener un baseline:** Construir un modelo complejo sin compararlo con un baseline simple. Si la regresión logística no supera a recomendar lo popular, no necesitas una red neuronal.
- **Medir solo métricas agregadas:** Un promedio puede esconder que el modelo funciona bien para el 90% de usuarios y pésimo para el 10%. Siempre mira distribuciones.

## Resumen

Los fundamentos de los sistemas de recomendación comienzan antes del primer algoritmo: entender el tipo de feedback (explícito vs implícito), elegir las métricas correctas (NDCG para ranking, CTR para online), reconocer los sesgos de evaluación (exposure bias, popularity bias), y siempre empezar con un baseline de popularidad.

El objetivo final no es maximizar una métrica, sino mejorar la experiencia del usuario. Un sistema con NDCG perfecto que recomienda siempre lo mismo aburre al usuario. Un buen sistema balancea precisión, diversidad, novedad y serendipia.

## Check Your Understanding

1. ¿Cuál es la principal desventaja del feedback explícito comparado con el implícito?
<!-- El feedback explícito es escaso: los usuarios no rating todo lo que consumen. El implícito es abundante pero ruidoso. -->

2. ¿Por qué NDCG es mejor que precision@k para evaluar rankings?
<!-- NDCG penaliza que un item relevante aparezca en posición baja. Precision@k trata igual un relevante en posición 1 que en posición k. -->

3. ¿Qué es exposure bias y cómo afecta la evaluación offline de recomendaciones?
<!-- Es el sesgo de que los datos offline reflejan lo que el sistema anterior mostró, no lo que el usuario hubiera preferido. Un item puede tener pocos clics porque nunca se mostró. -->

4. ¿Por qué un baseline de popularidad es un buen punto de partida?
<!-- Es simple, barato, efectivo en cold start, y establece una cota inferior para modelos más complejos. -->

5. ¿Cómo podrías mitigar el popularity bias en un sistema de recomendación?
<!-- Introduciendo aleatoriedad controlada (exploración), normalizando por popularidad, o usando técnicas de debiasing como IPW. -->

## Where to Go Next

- [[Collaborative Filtering]] — Recomendación basada en comportamiento de usuarios similares.
- [[Content-Based Filtering]] — Recomendación basada en atributos de los items.
- [[Hybrid & Session-Based]] — Combinación de enfoques y recomendación basada en sesión actual.
- [[Model Evaluation]] — Más métricas y técnicas de evaluación para ML.
- [[A-B Testing]] — Cómo comparar modelos en producción con experimentos controlados.
- [[Feature Engineering]] — Construcción de features para sistemas de recomendación.
- [[Statistics]] — Fundamentos estadísticos para entender biases y distribuciones.
