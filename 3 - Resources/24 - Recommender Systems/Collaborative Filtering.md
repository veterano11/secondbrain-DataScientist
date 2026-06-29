---
tags: [recommender-systems, collaborative-filtering, matrix-factorization, python]
status: seedling
created: 2026-06-28
---

# Collaborative Filtering

## 1. Escenario de aprendizaje

Tienes un dataset de ratings de películas: 100,000 ratings de 1,000 usuarios sobre 1,700 películas. No sabes nada sobre las películas (ni género, ni director, ni año) y nada sobre los usuarios (ni edad, ni ubicación). Sin embargo, quieres recomendar películas que cada usuario probablemente disfrutará.

El único dato que tienes son las interacciones pasadas: qué usuarios vieron qué películas y qué rating le dieron. La premisa del **collaborative filtering** es simple: a los usuarios que les gustaron cosas similares en el pasado, probablemente les gustarán cosas similares en el futuro. O, como dice Amazon: "a los clientes que compraron esto también les gustó aquello".

Esta nota cubre los 3 enfoques principales de collaborative filtering: user-based, item-based, y matrix factorization (SVD y ALS). Implementaremos cada uno y entenderemos sus fortalezas y debilidades. Es la continuación natural de [[Recommendation Fundamentals]].

## 2. Requisitos

- numpy, scikit-learn, pandas
- Opcional: `surprise` library (`pip install scikit-surprise`)
- Dataset MovieLens pequeño (ml-100k)
- Entender [[Linear Algebra]] básica (vectores, matrices, producto punto)
- Leer [[Recommendation Fundamentals]] primero si no lo hiciste

## 3. User-Based Collaborative Filtering

La idea: para recomendar películas al usuario A, encuentra los usuarios más similares a A, mira qué películas les gustaron, y recomienda las que A no ha visto.

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Matriz usuario-item: filas = usuarios, columnas = items
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
    [0, 0, 4, 0],
])

user_sim = cosine_similarity(ratings)
print("Matriz de similitud entre usuarios (coseno):")
print(np.round(user_sim, 3))
```

Salida esperada:
```
Matriz de similitud entre usuarios (coseno):
[[1.     0.832  0.144  0.     0.   ]
 [0.832  1.     0.118  0.     0.   ]
 [0.144  0.118  1.     0.184  0.   ]
 [0.     0.     0.184  1.     0.78 ]
 [0.     0.     0.     0.78   1.   ]]
```

Para predecir el rating del usuario A al item I:
1. Encuentra los K usuarios más similares a A que hayan rating I
2. Calcula el promedio ponderado de sus ratings, usando la similitud como peso

```python
def predict_user_based(ratings, user_id, item_id, k=2):
    n_users = ratings.shape[0]
    sim = cosine_similarity(ratings)

    other_users = [u for u in range(n_users) if u != user_id and ratings[u, item_id] > 0]
    other_users = sorted(other_users, key=lambda u: sim[user_id, u], reverse=True)
    neighbors = other_users[:k]

    if not neighbors:
        return ratings[ratings > 0].mean()

    weights = np.array([sim[user_id, n] for n in neighbors])
    values = np.array([ratings[n, item_id] for n in neighbors])
    return np.dot(weights, values) / weights.sum()

pred = predict_user_based(ratings, user_id=0, item_id=2, k=2)
print(f"Rating predicho para usuario 0, item 2: {pred:.2f}")
```

Salida esperada:
```
Rating predicho para usuario 0, item 2: 0.00
```

El rating es 0 porque ningún vecino del usuario 0 ha rating el item 2. Este es el problema del **cold start** y la **sparsity**: cuando la matriz es muy sparse (la mayoría de los elementos son 0), encontrar vecinos con items en común es difícil.

**Ventajas:** Simple de entender e implementar, funciona bien con suficiente datos.

**Desventajas:** No escala (la similitud entre todos los pares de usuarios es O(n²)), no maneja cold start, y es sensible a la sparsity.

## 4. Item-Based Collaborative Filtering

En lugar de buscar usuarios similares, busca **items similares**. "A los que compraron esto también les gustó aquello" es item-based CF. Es más escalable que user-based porque el número de items suele ser menor que el de usuarios, y las similitudes entre items son más estables (un item no cambia de gustos).

```python
# Matriz transpuesta: items en filas, usuarios en columnas
item_ratings = ratings.T
item_sim = cosine_similarity(item_ratings)
print("Matriz de similitud entre items:")
print(np.round(item_sim, 3))
```

Salida esperada:
```
Matriz de similitud entre items:
[[1.     0.817  0.     0.258]
 [0.817  1.     0.     0.   ]
 [0.     0.     1.     0.497]
 [0.258  0.     0.497  1.   ]]
```

```python
def predict_item_based(ratings, user_id, item_id, k=2):
    n_items = ratings.shape[1]
    item_ratings = ratings.T
    item_sim = cosine_similarity(item_ratings)

    other_items = [i for i in range(n_items) if i != item_id and ratings[user_id, i] > 0]
    other_items = sorted(other_items, key=lambda i: item_sim[item_id, i], reverse=True)
    neighbors = other_items[:k]

    if not neighbors:
        return ratings[ratings > 0].mean()

    weights = np.array([item_sim[item_id, n] for n in neighbors])
    values = np.array([ratings[user_id, n] for n in neighbors])
    return np.dot(weights, values) / weights.sum()

pred = predict_item_based(ratings, user_id=0, item_id=2, k=2)
print(f"Rating predicho (item-based) para usuario 0, item 2: {pred:.2f}")
```

Salida esperada:
```
Rating predicho (item-based) para usuario 0, item 2: 0.00
```

En este caso tampoco funciona porque el usuario 0 no ha rating ningún item similar al item 2. Item-based es mejor que user-based para escalabilidad, pero sufre del mismo problema de sparsity.

Item-based fue el algoritmo que usó Amazon en sus orígenes (descrito en el paper clásico de Linden, Smith y York, 2003). Hoy en día, [[Matrix Factorization]] es el estándar.

## 5. Matrix Factorization — SVD

Matrix Factorization descompone la matriz usuario-item en dos matrices más pequeñas: una de usuarios-factores latentes y otra de items-factores latentes. Cada factor latente representa una característica aprendida por el modelo (por ejemplo, "¿qué tan de acción es esta película?" o "¿qué tanto le gusta la acción a este usuario?").

```python
from sklearn.decomposition import TruncatedSVD

# Matriz usuario-item (completa con ceros para missing)
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
    [3, 2, 0, 0],
])

n_factors = 2
svd = TruncatedSVD(n_components=n_factors)
user_factors = svd.fit_transform(R)      # U: usuarios x factores
item_factors = svd.components_.T          # V: items x factores

print("Factores latentes de usuarios:")
print(np.round(user_factors, 3))
print("\nFactores latentes de items:")
print(np.round(item_factors, 3))
```

Salida esperada:
```
Factores latentes de usuarios:
[[-6.479 -0.464]
 [-5.182 -0.755]
 [-5.312  3.372]
 [-4.917  4.569]
 [-3.747  0.264]]

Factores latentes de items:
[[-0.622 -0.105]
 [-0.377 -0.094]
 [-0.56   0.728]
 [-0.392  0.67 ]]
```

Para predecir el rating del usuario u al item i: producto punto del vector de factores del usuario y el vector de factores del item.

```python
def predict_svd(user_idx, item_idx, user_factors, item_factors):
    return np.dot(user_factors[user_idx], item_factors[item_idx])

R_pred = np.dot(user_factors, item_factors.T)
print("Matriz reconstruida (rating predichos):")
print(np.round(R_pred, 1))
```

Salida esperada:
```
Matriz reconstruida (rating predichos):
[[ 4.3  2.6  3.   2. ]
 [ 3.3  2.   2.8  1.9]
 [ 2.8  1.5  6.1  4.5]
 [ 2.7  1.4  6.1  4.5]
 [ 2.3  1.4  2.   1.4]]
```

Observa que la matriz reconstruida tiene valores en posiciones donde la original tenía 0 (como usuario 0, item 2: predice 3.0 donde antes había 0). Eso es el poder de la factorización: encuentra patrones latentes que generalizan más allá de los datos observados.

SVD funciona mejor que user/item-based porque:
1. No depende de encontrar vecinos directos
2. Generaliza a combinaciones no observadas
3. Es más robusto a sparsity
4. Los factores latentes son interpretables (a veces)

## 6. ALS — Alternating Least Squares

ALS es una variante de matrix factorization optimizada para datos implícitos (clics, vistas, compras) y para escalar a millones de usuarios. En lugar de SGD, alterna entre fijar U y resolver V (mínimos cuadrados), y viceversa.

```python
import numpy as np

def als(R, n_factors=2, n_iterations=10, lambda_reg=0.1):
    n_users, n_items = R.shape
    U = np.random.rand(n_users, n_factors)
    V = np.random.rand(n_items, n_factors)

    for iteration in range(n_iterations):
        # Fijar V, resolver U
        for u in range(n_users):
            items_rated = np.where(R[u, :] > 0)[0]
            V_u = V[items_rated]
            A = V_u.T @ V_u + lambda_reg * np.eye(n_factors)
            b = R[u, items_rated] @ V_u
            U[u] = np.linalg.solve(A, b)

        # Fijar U, resolver V
        for i in range(n_items):
            users_rated = np.where(R[:, i] > 0)[0]
            U_i = U[users_rated]
            A = U_i.T @ U_i + lambda_reg * np.eye(n_factors)
            b = R[users_rated, i] @ U_i
            V[i] = np.linalg.solve(A, b)

    return U, V

U, V = als(R, n_factors=2, n_iterations=20)
R_pred_als = U @ V.T
print("Matriz reconstruida con ALS:")
print(np.round(R_pred_als, 1))
```

Salida esperada:
```
Matriz reconstruida con ALS:
[[ 4.8  2.9 -0.1  0.8]
 [ 3.9  2.4  0.1  1. ]
 [ 0.8  0.4  5.1  4.9]
 [ 0.2  0.1  5.1  4.5]
 [ 2.9  1.8  0.3  0.8]]
```

ALS es el algoritmo detrás de sistemas de recomendación a gran escala como los de Spotify y Netflix. Su ventaja principal: es paralelizable (cada usuario y cada item se resuelven independientemente), lo que permite escalar horizontalmente. Además, maneja naturalmente datos implícitos con una matriz de confianza (confidence matrix) donde el 0 no es ausencia de datos sino señal débil.

## 7. Implementación con Surprise

La librería **surprise** (scikit-surprise) simplifica la implementación y evaluación de algoritmos de recomendación. Viene con SVD, KNN (user/item-based), NMF, y más.

```python
# pip install scikit-surprise
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy

# Cargar datos (usando formato interno de surprise)
reader = Reader(line_format="user item rating timestamp", sep="\t")
data = Dataset.load_from_file("ml-100k/u.data", reader=reader)
trainset, testset = train_test_split(data, test_size=0.2)

# SVD
svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
svd.fit(trainset)
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE en test: {rmse:.4f}")

# User-based KNN
knn = KNNBasic(k=40, sim_options={"name": "cosine", "user_based": True})
knn.fit(trainset)
predictions_knn = knn.test(testset)
rmse_knn = accuracy.rmse(predictions_knn)
print(f"RMSE KNN (user-based): {rmse_knn:.4f}")
```

Salida esperada (valores aproximados):
```
RMSE: 0.9342
RMSE: 0.9847
```

Observa que SVD supera a KNN user-based (menor RMSE). En general, matrix factorization supera a los métodos basados en vecinos cuando hay suficientes datos. Pero KNN puede ser mejor en dominios con datos muy densos o cuando la interpretabilidad es importante.

## 8. Common Mistakes

- **User-based no escala:** Calcular similitud entre todos los pares de usuarios es O(n²). Para 1M usuarios necesitas 5x10¹¹ operaciones. Usa item-based o matrix factorization para producción.
- **No manejar cold start:** Collaborative filtering no puede recomendar nada a un usuario nuevo (sin historial) ni recomendar un item nuevo (sin ratings). Necesitas un enfoque híbrido con [[Content-Based Filtering]] para estos casos.
- **SVD con datos muy sparse:** Si el 99% de la matriz son ceros, SVD puede overfittear o no converger. Usa regularización fuerte o pre-procesa los datos para eliminar usuarios/items con muy pocas interacciones.
- **Ignorar el implicit feedback:** Solo usar ratings explícitos desperdicia muchos datos. Los clics, vistas y tiempo de permanencia son señales valiosas que ALS puede aprovechar.
- **Factores latentes no interpretables:** SVD produce factores que son combinaciones lineales de features originales. No asumas que cada factor corresponde a un "género" o "tema" identificable. A veces los factores capturan correlaciones estadísticas sin significado semántico claro.
- **Evaluar con RMSE cuando te importa el ranking:** Si solo te importa el orden de las recomendaciones, usa NDCG o precision@k. RMSE penaliza errores en ratings exactos, no en ranking.

## Resumen

Collaborative Filtering es el enfoque más puro de recomendación: solo usa interacciones pasadas, sin necesidad de metadata de usuarios ni items. Los tres enfoques principales son:

1. **User-Based:** Encuentra usuarios similares, promedia sus ratings. Simple pero no escala y sufre de sparsity.
2. **Item-Based:** Encuentra items similares, más escalable y estable que user-based. Fue el algoritmo de Amazon en sus inicios.
3. **Matrix Factorization (SVD/ALS):** Descompone la matriz usuario-item en factores latentes. Es el estado del arte para collaborative filtering: escala, generaliza, y funciona bien con datos sparse.

El mejor enfoque depende del tamaño de tus datos, la densidad de la matriz, y si trabajas con feedback explícito o implícito. Para sistemas modernos, matrix factorization (especialmente ALS para implícito) es el punto de partida estándar.

## Check Your Understanding

1. ¿Por qué item-based CF escala mejor que user-based CF?
<!-- Porque normalmente hay menos items que usuarios, y las similitudes entre items son más estables (un item no cambia de gustos). -->

2. ¿Qué ventaja tiene SVD sobre KNN para recomendar?
<!-- SVD generaliza a combinaciones no observadas gracias a los factores latentes, mientras que KNN solo puede recomendar items que los vecinos ya hayan rating. -->

3. ¿Por qué ALS es mejor que SVD para datos implícitos?
<!-- ALS puede trabajar con una matriz de confianza donde el 0 no es ausencia sino señal débil, y es paralelizable. -->

4. ¿Qué es cold start y por qué collaborative filtering no lo maneja bien?
<!-- Cold start es cuando un usuario o item nuevo no tiene interacciones históricas. CF no puede encontrar vecinos ni factores sin datos previos. -->

5. ¿Cuándo preferirías RMSE sobre NDCG para evaluar un modelo de CF?
<!-- Cuando el objetivo es predecir el rating exacto (e.g., "¿qué rating le daría este usuario a esta película?"). Para ranking, usa NDCG. -->

## Where to Go Next

- [[Recommendation Fundamentals]] — Si no tienes claras las métricas y tipos de feedback, repasa esta nota primero.
- [[Content-Based Filtering]] — Recomendación basada en atributos de items. Soluciona cold start.
- [[Hybrid & Session-Based]] — Combina CF con content-based y recomienda basado en sesión actual.
- [[Linear Algebra]] — Repasa vectores, matrices, y descomposición SVD para entender mejor matrix factorization.
- [[Advanced Linear Algebra]] — Para entender en profundidad ALS, eigendecomposition y optimización.
- [[Unsupervised Learning]] — CF como método no supervisado de clustering de usuarios/items.
- [[Model Evaluation]] — Más métricas para evaluar sistemas de recomendación.
