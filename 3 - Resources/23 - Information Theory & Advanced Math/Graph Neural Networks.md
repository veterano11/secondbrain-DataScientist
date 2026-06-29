---
tags:
  - graph-neural-networks
  - deep-learning
  - advanced
status: seedling
created: 2026-06-28
---

# Graph Neural Networks

## 1. Escenario de aprendizaje

Trabajas en un equipo de química computacional. Tu empresa sintetiza moléculas candidatas a fármacos y necesita predecir si una molécula será tóxica antes de invertir en ensayos clínicos. Cada molécula es un grafo: átomos son nodos, enlaces son aristas.

Tu primer intento es extraer features manuales: contar átomos de cada elemento, peso molecular, número de anillos. Entrenas un [[Supervised Learning]] clásico (Random Forest) y obtienes AUC de 0.72.

Un paper reciente reporta AUC de 0.94 usando Graph Neural Networks (GNNs). El problema no es tu modelo — es que perdiste la estructura: dos moléculas pueden tener los mismos átomos pero una ser tóxica y la otra no, porque los patrones de conectividad importan.

Esta nota cubre GNNs: cómo aprenden representaciones de [[Graph Theory & Network Analysis]] usando [[Neural Networks]] que respetan la topología del grafo.

---

## 2. ¿Qué problema resuelven las GNNs?

### 2.1 ML clásico en grafos

Para aplicar ML a grafos, necesitábamos feature engineering manual: estadísticos del grafo (diámetro, clustering coefficient), descriptores moleculares, counts de subgrafos (graphlets). Esto es costoso, no generaliza, y pierde información.

### 2.2 La propuesta GNN

Las GNNs aprenden representaciones (embeddings) de nodos y grafos directamente de la estructura, usando **message passing**: cada nodo agrega información de sus vecinos, actualiza su representación, y repite. Después de suficientes rondas, el embedding de cada nodo codifica su vecindario.

Tipos de tareas:

- **Clasificación de nodos:** predecir propiedad de un nodo (fraude en transacciones)
- **Clasificación de grafos:** predecir propiedad de todo el grafo (toxicidad molecular)
- **Predicción de enlaces:** predecir si existe conexión (recomendación)

### 2.3 Notación

Sea `G = (V, E)` un grafo con `N = |V|` nodos. `X ∈ ℝ^(N×F)` es la matriz de features de nodos. `A ∈ {0,1}^(N×N)` es la matriz de adyacencia. Queremos aprender `h_v ∈ ℝ^d` para cada nodo `v`.

---

## 3. Message Passing

### 3.1 Marco general

Cada capa GNN realiza tres pasos:

1. **Message:** cada vecino `u` envía un mensaje a `v` basado en sus embeddings
2. **Aggregate:** `v` agrega todos los mensajes de sus vecinos (suma, media, max)
3. **Update:** `v` combina su embedding actual con el agregado para producir el nuevo embedding

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({ MESSAGE(h_u^(k)) : u ∈ N(v) }))
```

### 3.2 Ejemplo simple

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMessagePassing(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, h, adj):
        # h: (N, F), adj: (N, N)
        N = h.shape[0]

        # Message de cada vecino
        messages = []

        for v in range(N):
            neighbors = adj[v].nonzero(as_tuple=True)[0]
            neighbor_msgs = h[neighbors]  # (num_neighbors, F)
            # Aggregate: sum
            agg = neighbor_msgs.sum(dim=0) if len(neighbor_msgs) > 0 else torch.zeros(h.shape[1])
            # Concatenate with current node embedding
            combined = torch.cat([h[v], agg], dim=0)
            messages.append(combined)

        messages = torch.stack(messages)
        return self.linear(messages)
```

**Nota:** Esta implementación es pedagógica y no escalable (loop explícito). Las implementaciones reales usan operaciones dispersas (sparse matrices).

---

## 4. GCN — Graph Convolutional Network

### 4.1 Convolución sobre vecinos

La GCN (Kipf & Welling, 2017) normaliza la agregación por grado para evitar que nodos con muchos vecinos dominen:

```
h_v^(k+1) = σ( Σ_{u ∈ N(v) ∪ {v}} (1 / sqrt(deg(v) * deg(u))) * W * h_u^(k) )
```

### 4.2 Implementación

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # edge_index: (2, E) — lista de aristas en formato COO
        return self.conv(x, edge_index)

# Clasificador de nodos con 2 capas GCN
class TwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 4.3 Self-loops

GCN agrega automáticamente conexiones a sí mismo (self-loop) para que el embedding del nodo actual también se considere en la agregación. Esto es crítico: sin self-loop, cada capa solo mira vecinos, perdiendo la propia identidad del nodo.

---

## 5. GAT — Graph Attention Network

### 5.1 Atención sobre vecinos

GAT (Veličković et al., 2018) asigna pesos de atención aprendidos a cada vecino, en lugar de promedios fijos. Cada nodo puede enfocarse en vecinos más relevantes.

```
α_uv = softmax_u( LeakyReLU( a^T [ W h_u || W h_v ] ) )

h_v' = σ( Σ_{u ∈ N(v)} α_uv * W * h_u )
```

### 5.2 Multi-head attention

```python
from torch_geometric.nn import GATConv

class GATClassifier(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_features, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_features * heads, num_classes, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Ejemplo de forward
model = GATClassifier(in_features=100, hidden_features=8, num_classes=2)
x = torch.randn(1000, 100)  # 1000 nodos, 100 features
edge_index = torch.randint(0, 1000, (2, 5000))  # 5000 aristas
out = model(x, edge_index)
print(out.shape)
```

**Salida esperada:**

```
torch.Size([1000, 2])
```

Cada nodo produce una distribución de probabilidad sobre 2 clases.

---

## 6. GraphSAGE — Sample and Aggregate

### 6.1 Escalabilidad

GCN y GAT requieren el vecindario completo de cada nodo. En grafos con miles de millones de nodos (red social de Facebook), esto es inviable. GraphSAGE (Hamilton et al., 2017) **samplea** un número fijo de vecinos por nodo.

### 6.2 Arquitectura

```python
from torch_geometric.nn import SAGEConv

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden_features)
        self.conv2 = SAGEConv(hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 6.3 Funciones de agregación

GraphSAGE soporta múltiples agregadores:

- **Mean:** promedio de embeddings de vecinos
- **Pool:** max-pooling sobre MLP aplicado a cada vecino
- **LSTM:** sobre secuencia ordenada de vecinos (requiere ordenación)

```python
# Diferentes agregadores en SAGEConv
conv_mean = SAGEConv(in_features, out_features, aggr='mean')
conv_max = SAGEConv(in_features, out_features, aggr='max')
conv_lstm = SAGEConv(in_features, out_features, aggr='lstm')
```

---

## 7. Pooling — Del grafo a un embedding global

Para clasificación de grafos completos (como la toxicidad molecular), necesitamos reducir los embeddings de todos los nodos a un solo vector. Esto es **graph pooling** o **readout**.

### 7.1 Pooling global

```python
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch

# Después de N capas GNN, todos los nodos tienen embeddings
# batch vector indica a qué grafo pertenece cada nodo
class GraphClassifier(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.classifier = nn.Linear(hidden_features, num_classes)

    def forward(self, x, edge_index, batch):
        # batch: tensor de longitud N, batch[i] = índice del grafo del nodo i
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling: embedding del grafo completo
        x = global_mean_pool(x, batch)  # (num_graphs, hidden_features)
        return self.classifier(x)
```

**Salida esperada:** Para 32 moléculas en un batch, output de shape `(32, num_classes)`.

### 7.2 Efecto del pooling

- **Mean pool:** sensible a la distribución, no al tamaño del grafo
- **Sum pool:** sensible al tamaño del grafo (moléculas más grandes tienen suma mayor)
- **Max pool:** captura la característica más prominente, ignora el resto

En la práctica, mean pool es el default, pero la elección depende del problema.

---

## 8. Aplicaciones principales

### 8.1 Predicción de enlaces (recomendación)

Dado un grafo parcial (usuarios, items, interacciones conocidas), predecir aristas faltantes. Clásico en [[Recommender Systems]].

```python
# Codificar nodos con GNN
z = gnn_encoder(x, edge_index)  # (N, d)

# Score de enlace entre nodo i y nodo j
score = torch.sigmoid((z[i] * z[j]).sum(dim=-1))
```

### 8.2 Clasificación de nodos (detección de fraude)

Cada transacción es un nodo en un grafo de transacciones. Vecinos = transacciones relacionadas (misma tarjeta, mismo IP). GNN propaga información de nodos fraudulentos conocidos a nodos sospechosos.

### 8.3 Clasificación de grafos (química)

Como en el escenario inicial: predecir propiedades moleculares (toxicidad, solubilidad, actividad biológica) desde la estructura molecular representada como grafo.

### 8.4 Modelado de física y ciencias

- Simulación de dinámica de partículas (interacciones como aristas)
- Descubrimiento de fármacos (generación de moléculas con GNN + VAE)
- Predicción de estructura de proteínas (AlphaFold usa principios similares)

---

## 9. Common Mistakes

### No considerar self-loops
- **Error:** No agregar conexión del nodo a sí mismo; cada capa solo mira vecinos, ignorando el embedding actual
- **Solución:** Usar capas que incluyan self-loops automáticamente (GCNConv, GATConv) o agregarlos manualmente

### Oversmoothing con muchas capas
- **Error:** Apilar 10+ capas GNN esperando capturar vecindarios lejanos; todos los embeddings convergen a un valor similar
- **Solución:** Limitar a 2-4 capas, o usar técnicas anti-oversmoothing (PairNorm, DropEdge, JK Connections)

### Asumir grafos estáticos cuando son dinámicos
- **Error:** Entrenar una GNN en un snapshot del grafo cuando las aristas cambian en el tiempo (red social, transacciones)
- **Solución:** Usar GNNs temporales (TGAT, EvolveGCN) que modelan la evolución temporal del grafo

### No normalizar features de nodos
- **Error:** Features de nodos con escalas muy diferentes (ej. peso atómico y número atómico) sin escalar
- **Solución:** Normalizar features de nodos antes de alimentar la GNN (LayerNorm, BatchNorm)

### Ignorar features de aristas
- **Error:** En química, los tipos de enlace (simple, doble, aromático) son críticos; GNN básica solo usa conectividad
- **Solución:** Usar capas que acepten edge features (GINE, EGNN) o codificar edge features como atributos

### Tratar el batch incorrectamente
- **Error:** En clasificación de grafos, no usar batch vector, mezclando nodos de diferentes moléculas en pooling
- **Solución:** Pasar `batch` vector a todas las capas de pooling. PyTorch Geometric lo maneja automáticamente

---

## Resumen

| Arquitectura | Agregación | Escalabilidad | Mejor para |
|---|---|---|---|
| GCN | Promedio normalizado por grado | Media (grafo completo) | Grafos homogéneos, tamaño moderado |
| GAT | Atención aprendida | Media (grafo completo) | Grafos donde vecinos tienen relevancia variable |
| GraphSAGE | Sample + Aggregate | Alta (samplea vecinos) | Grafos masivos, producción |
| GIN | Suma + MLP | Media | Poder expresivo máximo, clasificación de grafos |

Las GNNs resuelven el problema de aprender representaciones de grafos sin feature engineering manual. La arquitectura correcta depende del tamaño del grafo, la tarea y si las relaciones entre nodos son homogéneas o requieren atención diferencial.

---

## Check Your Understanding

**1. ¿Qué problema resuelve el self-loop en GCN?**
<!-- Sin self-loop, el embedding de un nodo en la capa k+1 depende solo de sus vecinos, no de su propio embedding en la capa k. Self-loop permite que el nodo se considere a sí mismo en la agregación. -->

**2. ¿Por qué GAT es más expresivo que GCN?**
<!-- GCN asigna pesos fijos (1/grado) a cada vecino. GAT aprende pesos de atención, permitiendo que el modelo decida qué vecinos son más relevantes para cada nodo. -->

**3. ¿Qué causa oversmoothing y cómo evitarlo?**
<!-- Con muchas capas, los embeddings de todos los nodos convergen a un vector similar porque cada capa promedia vecinos. Se evita limitando profundidad a 2-4 capas o usando técnicas como DropEdge, PairNorm, JK Connections. -->

**4. ¿Cuándo preferirías GraphSAGE a GCN?**
<!-- Cuando el grafo es masivo (millones de nodos) y no cabe en memoria. GraphSAGE samplea vecinos, permitiendo entrenamiento por mini-batches. GCN requiere el grafo completo. -->

**5. ¿Qué es el pooling en clasificación de grafos y por qué necesitamos batch vector?**
<!-- Pooling reduce embeddings de nodos a un embedding global por grafo. batch vector indica qué nodos pertenecen a cada grafo en el batch, necesario para separar grafos durante el pooling. -->

---

## Where to Go Next

- [[Graph Theory & Network Analysis]] — Conceptos fundamentales: grado, caminos, conectividad, clustering
- [[Neural Networks]] — Base de toda arquitectura GNN: retropropagación, funciones de activación, dropout
- [[Advanced Linear Algebra]] — Multiplicación de matrices dispersas, diagonalización, eigenvectores de grafos
- [[Deep Learning]] — Técnicas avanzadas de entrenamiento, normalización, optimización
- [[Word Embeddings (Word2Vec)]] — Inspiración para aprender embeddings de nodos (Node2Vec, DeepWalk)
- [[Unsupervised Learning]] — Autoencoders de grafos, GraphVAE, aprendizaje de representaciones sin etiquetas
- [[Recommender Systems]] — Aplicación clásica de predicción de enlaces con GNNs
