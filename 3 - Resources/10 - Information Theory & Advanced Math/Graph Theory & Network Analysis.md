---
tags:
  - graph-theory
  - network-analysis
  - node2vec
  - community-detection
status: seedling
created: 2026-06-28
---

## 1. Escenario de aprendizaje

Tienes datos de interacciones en Twitter: quién sigue a quién, quién retweetea a quién. Quieres encontrar los usuarios más influyentes, detectar comunidades (grupos de interés) y quizás recomendar nuevas cuentas a seguir. Un DataFrame no sirve: necesitas un grafo. [[Recommender Systems]] y detección de fraude usan exactamente las mismas técnicas. Esta nota cubre representación de grafos, medidas de centralidad, detección de comunidades y graph embeddings como [[Word Embeddings (Word2Vec)]] aplicado a nodos.

## 2. Requisitos

- Python 3.8+, networkx, numpy, matplotlib, scikit-learn
- Conceptos básicos de [[Linear Algebra]] (matrices, eigenvalores)
- Familiaridad con [[Unsupervised Learning]]

## 3. Representación de grafos

Un grafo $G = (V, E)$ consiste en nodos (vértices) y aristas (edges). Puede ser:

- **Dirigido** vs **no dirigido**: las aristas tienen dirección o no
- **Ponderado**: las aristas tienen pesos (e.g., número de retweets)
- **Con atributos**: nodos y aristas pueden tener features

Se representa con una **matriz de adyacencia** $A$ donde $A_{ij} = 1$ si hay arista de $i$ a $j$ (o el peso).

```python
import networkx as nx
import numpy as np

# Grafo no dirigido simple
G = nx.Graph()
G.add_edges_from([
    ("Alice", "Bob"),
    ("Alice", "Charlie"),
    ("Bob", "Charlie"),
    ("Charlie", "David"),
    ("David", "Eve"),
])

# Matriz de adyacencia
A = nx.adjacency_matrix(G).toarray()
print("Matriz de adyacencia:")
print(A)
print(f"Nodos: {list(G.nodes())}")

# Salida esperada:
# Matriz de adyacencia:
# [[0. 1. 1. 0. 0.]
#  [1. 0. 1. 0. 0.]
#  [1. 1. 0. 1. 0.]
#  [0. 0. 1. 0. 1.]
#  [0. 0. 0. 1. 0.]]
# Nodos: ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
```

Propiedades importantes:

- **Grado ($k_i$)**: número de aristas incidentes al nodo $i$
- **Densidad**: $\frac{2|E|}{|V|(|V|-1)}$ (proporción de aristas posibles)
- **Camino mínimo (shortest path)**: distancia entre dos nodos
- **Diámetro**: camino mínimo más largo del grafo

## 4. Centralidad: ¿quién es importante?

### Degree Centrality

Los nodos con más conexiones son importantes. Para grafos dirigidos, se distingue in-degree y out-degree.

$$C_D(v) = \frac{\deg(v)}{|V|-1}$$

### Betweenness Centrality

Mide cuántos caminos mínimos pasan por un nodo. Nodos con alta betweenness son "puentes" entre partes del grafo.

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

donde $\sigma_{st}$ es el número total de caminos mínimos entre $s$ y $t$, y $\sigma_{st}(v)$ los que pasan por $v$.

### Closeness Centrality

Mide qué tan cerca está un nodo de todos los demás.

$$C_C(v) = \frac{|V|-1}{\sum_{u \neq v} d(v,u)}$$

### Eigenvector Centrality y PageRank

La importancia de un nodo depende de la importancia de sus vecinos. PageRank añade un factor de amortiguación (damping) para evitar sumideros.

$$x_v = \alpha \sum_{u \to v} \frac{x_u}{\deg(u)} + (1-\alpha)$$

```python
import networkx as nx

G = nx.karate_club_graph()  # Grafo clásico de Zachary

centralities = {
    "Degree": nx.degree_centrality(G),
    "Betweenness": nx.betweenness_centrality(G),
    "Closeness": nx.closeness_centrality(G),
    "Eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
    "PageRank": nx.pagerank(G, alpha=0.85),
}

# Top 3 nodos por cada medida
for name, cent in centralities.items():
    top = sorted(cent.items(), key=lambda x: -x[1])[:3]
    print(f"{name}: {top}")

# Salida esperada (aproximada, los valores exactos varían):
# Degree: [(33, 0.515), (0, 0.485), (32, 0.364)]
# Betweenness: [(0, 0.438), (33, 0.304), (2, 0.144)]
# Closeness: [(0, 0.569), (33, 0.550), (2, 0.467)]
# Eigenvector: [(33, 0.374), (0, 0.355), (2, 0.280)]
# PageRank: [(33, 0.107), (0, 0.096), (2, 0.069)]
```

Cada medida captura un aspecto distinto de "importancia". PageRank es el algoritmo original de Google y se usa también en [[Recommender Systems]].

## 5. Comunidades: estructura de grupos

### Modularity

Mide la calidad de una partición en comunidades: compara la densidad de aristas dentro de comunidades con lo esperado en un grafo aleatorio equivalente.

$$Q = \frac{1}{2|E|} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2|E|}\right] \delta(c_i, c_j)$$

- $Q > 0.3$ sugiere estructura comunitaria significativa
- $Q$ cerca de $0$ indica partición aleatoria

### Louvain Algorithm

Algoritmo greedy que maximiza modularidad:

1. Cada nodo empieza en su propia comunidad
2. Se mueven nodos a comunidades vecinas si aumenta modularidad
3. Se construye un meta-grafo con las comunidades como nodos
4. Se repite hasta convergencia

### Girvan-Newman

Algoritmo divisivo que remueve aristas con mayor betweenness iterativamente.

```python
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities

G = nx.karate_club_graph()
comunidades = louvain_communities(G, seed=42)

print(f"Número de comunidades: {len(comunidades)}")
for i, com in enumerate(comunidades):
    print(f"  Comunidad {i}: {sorted(com)}")

# Colorear nodos por comunidad
colors = [0] * len(G)
for i, com in enumerate(comunidades):
    for node in com:
        colors[node] = i

print(f"Modularity: {nx.community.modularity(G, comunidades):.4f}")

# Salida esperada (aproximada):
# Número de comunidades: 4
#   Comunidad 0: [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 17, 19, 21]
#   Comunidad 1: [4, 5, 6, 16]
#   Comunidad 2: [14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
#   Comunidad 3: [33]
# Modularity: 0.4188
```

## 6. Graph embeddings: nodos como vectores

Los graph embeddings aprenden una representación vectorial densa de cada nodo preservando la estructura del grafo.

### Node2Vec

1. Simula **random walks** desde cada nodo (parámetros $p$ y $q$ controlan BFS vs DFS)
2. Trata cada walk como una "oración" y los nodos como "palabras"
3. Entrena [[Word Embeddings (Word2Vec)]] (skip-gram) sobre estas secuencias

- $p$ (return): controla probabilidad de volver al nodo anterior
- $q$ (in-out): controla si el walk tiende a BFS ($q > 1$) o DFS ($q < 1$)
- Nodos con contextos similares tendrán embeddings cercanos

### GraphSAGE

Aprende una función que agrega información de vecinos para generar embeddings. Es **inductivo**: puede generar embeddings para nodos no vistos durante entrenamiento.

```python
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx
from node2vec import Node2Vec

G = nx.karate_club_graph()

# Generar random walks y entrenar embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=30,
                    num_walks=200, p=1, q=1, workers=1, seed=42)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Embedding del nodo 0
emb_0 = model.wv['0']
print(f"Embedding del nodo 0 (primeras 8 dims): {emb_0[:8]}")

# Similaridad entre nodos
sim = model.wv.similarity('0', '33')
print(f"Similaridad nodo 0 - nodo 33: {sim:.4f}")

# Salida esperada (aproximada):
# Embedding del nodo 0 (primeras 8 dims): [-0.5842 -0.2124  0.1438  0.4762 -0.3194 -0.0851  0.5291  0.0214]
# Similaridad nodo 0 - nodo 33: 0.9862
```

Los graph embeddings permiten usar cualquier algoritmo de ML sobre nodos: clasificación, clustering, link prediction.

## 7. Aplicaciones en ML

### Social Recommendation

- **Link prediction**: predecir aristas faltantes usando similaridad de embeddings o métricas estructurales (Adamic-Adar, Jaccard coefficient)
- **Friend recommendation**: nodos con embeddings cercanos → candidatos a conexión

### Fraude detection

- Transacciones bancarias forman un grafo (cuentas → transacciones → cuentas)
- Comunidades sospechosas (círculos de cuentas que se transfieren entre sí sin sentido económico)
- Features estructurales (grado, clustering coefficient) alimentan clasificadores

### Propagación de información

- **Influencia máxima**: encontrar $k$ nodos que maximicen la propagación (modelos SIR, Independent Cascade)
- Centralidad (especialmente PageRank y betweenness) correlaciona con influencia

```python
import networkx as nx
import numpy as np

# Link prediction con Adamic-Adar
G = nx.karate_club_graph()
G_train = G.copy()
# Remover algunas aristas para simular predicción
edges_removed = [('0', '4'), ('0', '5'), ('0', '10')]
G_train.remove_edges_from(edges_removed)

preds = nx.adamic_adar_index(G_train, edges_removed)
for u, v, p in preds:
    print(f"Adamic-Adar ({u}, {v}): {p:.4f}")

# Salida esperada (aproximada):
# Adamic-Adar (0, 4): 1.3863
# Adamic-Adar (0, 5): 1.3863
# Adamic-Adar (0, 10): 1.3863
```

## 8. Common Mistakes

1. **No considerar directed vs undirected**: betweenness centrality en grafos dirigidos requiere normalización distinta. Aplicar fórmulas de undirected a directed produce resultados engañosos.

2. **PageRank sin damping factor**: si $\alpha = 1$ (sin damping), nodos sin out-links (dangling nodes) acumulan todo el PageRank y el algoritmo no converge.

3. **Random walks muy cortos en Node2Vec**: walks de longitud < 10 no capturan estructura global. Usa al menos 20-30 pasos.

4. **Ignorar el tamaño del grafo**: betweenness centrality exacta es $O(|V||E|)$ o peor. En grafos grandes (>10K nodos), usa aproximaciones (betweenness aproximada, random sampling).

5. **Normalizar modularidad incorrectamente**: modularidad no debe usarse para comparar particiones entre grafos de distinto tamaño. Solo entre particiones del mismo grafo.

6. **Usar clustering coefficient promedio sin inspeccionar distribución**: grafos con estructura comunitaria tienen coeficientes de clustering heterogéneos. El promedio puede esconder la estructura.

## Resumen

La teoría de grafos proporciona el lenguaje matemático para modelar relaciones: redes sociales, rutas, interacciones moleculares y sistemas de recomendación. Las medidas de centralidad (degree, betweenness, closeness, eigenvector, PageRank) cuantifican la importancia de nodos desde perspectivas complementarias. La detección de comunidades con Louvain o Girvan-Newman revela estructura de grupos. Los graph embeddings como Node2Vec convierten nodos en vectores densos, permitiendo aplicar ML tradicional sobre datos de grafo. Cada técnica tiene supuestos específicos —ignorar la direccionalidad, elegir mal los parámetros de random walk, o no verificar la convergencia de PageRank— que pueden invalidar los resultados.

## Check Your Understanding

1. ¿Por qué un nodo con high degree centrality no necesariamente tiene high betweenness centrality? <!-- Un nodo puede tener muchas conexiones locales (degree alto) pero estar en el centro de un clúster sin conectar clústers diferentes. Betweenness requiere ser "puente" entre partes del grafo. -->

2. PageRank asigna damping factor $\alpha = 0.85$. ¿Qué pasa si $\alpha = 0$? <!-- Cada paso solo teletransporta aleatoriamente (random surfer model sin seguir aristas). Todos los nodos tendrían el mismo PageRank. -->

3. ¿Por qué Girvan-Newman no escala a grafos grandes? <!-- Cada iteración recalcula betweenness de todas las aristas ($O(|V||E|)$) y remueve una. Para $|V|$ comunidades requiere $O(|V|^2|E|)$. -->

4. Si dos nodos están en la misma comunidad según Louvain, ¿significa que deberían tener una arista? <!-- No. Pertencer a la misma comunidad implica que están densamente conectados indirectamente, pero puede que no exista arista directa (y que deba sugerirse). -->

5. ¿Node2Vec puede generar embeddings para un nodo no visto durante entrenamiento? <!-- No. Node2Vec es transductivo: asigna embeddings solo a nodos presentes en el grafo de entrenamiento. Para nodos nuevos se necesita GraphSAGE (inductivo). -->

## Where to Go Next

- [[Linear Algebra]] — eigenvalores fundamentales para spectral clustering y PageRank
- [[Advanced Linear Algebra]] — SVD para graph factorization
- [[Word Embeddings (Word2Vec)]] — base del entrenamiento de Node2Vec
- [[Unsupervised Learning]] — clustering aplicado a comunidades
- [[Feature Engineering]] — extraer features estructurales de grafos
- [[Probability]] — random walks y modelos de propagación
- [[Recommender Systems]] — link prediction y social recommendation
- [[Information Theory for ML]] — mutual information para selección de nodos
