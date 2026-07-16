---
tags: [machine-learning, unsupervised, core]
status: growing
created: 2026-06-27
---

# Aprendizaje No Supervisado

## 1. Escenario de aprendizaje

Trabajas en una empresa de comercio electrónico con millones de clientes. Tienes datos de compras, navegación y soporte técnico, pero no tienes etiquetas — nadie te ha dicho "este cliente es de tipo A, este es de tipo B". Necesitas encontrar segmentos de clientes por su comportamiento para personalizar ofertas y campañas de marketing. Sin etiquetas disponibles, el aprendizaje no supervisado descubre **estructura oculta en los datos**: agrupa clientes similares, reduce dimensionalidad para visualización y detecta anomalías como fraude.

Tres grandes tareas no supervisadas:
- **Clustering**: agrupar ítems similares
- **Reducción de dimensionalidad**: comprimir datos preservando estructura
- **Detección de anomalías**: encontrar patrones inusuales

Estas son esenciales para el análisis exploratorio de datos, feature engineering y entender tus datos antes de aplicar métodos supervisados.

---

## 2. Clustering

### 2.1 K-Means

**Objetivo**: particionar $n$ puntos en $k$ clusters, minimizando la varianza dentro de cada cluster.

**El algoritmo** (paso a paso):
1. Elegir $k$ puntos aleatorios como centroides iniciales
2. **Asignar**: cada punto va al centroide más cercano
3. **Actualizar**: recalcular centroides como la media de los puntos en cada cluster
4. Repetir pasos 2-3 hasta convergencia (los centroides dejan de cambiar)

**Ejemplo concreto** — clustering de clientes:
```
Clientes: (edad, ingresos) = [(25, 30k), (30, 35k), (55, 80k), (60, 85k)]
k = 2

Paso 1: centroide1 = (25, 30k), centroide2 = (55, 80k)  (inicialización aleatoria)
Paso 2: cliente (30, 35k) → centroide1 (más cercano)
         cliente (60, 85k) → centroide2 (más cercano)
Paso 3: centroide1 = ((25+30)/2, (30+35)/2) = (27.5, 32.5k)
         centroide2 = ((55+60)/2, (80+85)/2) = (57.5, 82.5k)
Repetir → converge en pocas iteraciones
```

**Elección de $k$**:
- **Método del codo**: graficar inercia (varianza dentro del cluster) vs $k$. Buscar el "codo" donde agregar más clusters da rendimientos decrecientes.
- **Score de silueta**: mide qué tan similares son los puntos a su propio cluster vs otros clusters. Mayor es mejor.

**Limitaciones**:
- Necesita especificar $k$ de antemano
- Sensible a la inicialización (k-means++ ayuda)
- Asume clusters esféricos de tamaño similar
- Tiene problemas con formas no convexas, densidades variables

### 2.2 DBSCAN

**Clustering basado en densidad**: los clusters son regiones de alta densidad separadas por regiones de baja densidad.

**Cómo funciona**:
1. Para cada punto, contar vecinos dentro del radio $\epsilon$
2. Si vecinos $\geq$ minPts, el punto es un **punto central** (inicia un cluster)
3. Todos los puntos dentro de $\epsilon$ de un punto central pertenecen al mismo cluster
4. Puntos no alcanzables desde ningún punto central son **ruido** (outliers)

**Ventajas sobre K-Means**:
- No requiere $k$
- Puede encontrar clusters de formas arbitrarias
- Detecta outliers automáticamente
- Robusto al ruido

**Desventajas**:
- Sensible a los parámetros $\epsilon$ y minPts
- Tiene problemas con densidades variables
- No funciona bien en dimensiones altas (maldición de la dimensionalidad)

### 2.3 Clustering Jerárquico

Construye un árbol (dendrograma) de clusters.

**Aglomerativo** (de abajo hacia arriba, más común):
1. Inicio: cada punto es su propio cluster
2. Fusionar los dos clusters más cercanos
3. Repetir hasta que quede un solo cluster

**Divisivo** (de arriba hacia abajo): empezar con un cluster, dividir recursivamente.

El dendrograma te permite elegir el número de clusters después — solo corta el árbol en cualquier nivel.

---

## 3. Reducción de Dimensionalidad

### 3.1 Análisis de Componentes Principales (PCA)

**Objetivo**: encontrar una representación de menor dimensión que preserve la varianza máxima.

**Intuición**: imagina una nube de puntos en 3D con forma de panqueque plano. La mayor parte de la varianza (información) está en el plano del panqueque; el grosor es ruido. PCA encuentra el plano.

**Cómo funciona** (mediante SVD):
1. Centrar los datos: restar la media de cada feature
2. Calcular $X = U \Sigma V^T$ (SVD de los datos centrados)
3. Componentes principales = columnas de $V$ (direcciones de máxima varianza; ver [[Linear Algebra]] para detalles de SVD)
4. Proyecciones = $U_k \Sigma_k$ (datos en espacio reducido)

**Elección del número de componentes**:
- Ver la razón de varianza explicada: $\frac{\sigma_i^2}{\sum \sigma_j^2}$ (ver [[Statistics]] para conceptos de varianza)
- Elegir $k$ tal que la varianza acumulada > 0.9 o 0.95

**Cuándo usar PCA**:
- Visualización (proyectar a 2D o 3D)
- Reducción de ruido (componentes pequeños suelen ser ruido)
- Acelerar entrenamiento (menos features)
- Antes de algoritmos basados en distancia (KNN, K-Means) en dimensiones altas

**Advertencia**: PCA asume estructura lineal. No captura relaciones no lineales.

### 3.2 t-SNE

**Objetivo**: visualizar datos de alta dimensionalidad en 2D o 3D preservando la estructura local.

**Por qué es diferente de PCA**:
- PCA preserva la varianza global (estructura a gran escala)
- t-SNE preserva vecindarios locales (puntos cercanos se mantienen cercanos)

t-SNE es no determinista, costoso (O(n²)), y debe usarse **solo para visualización**, no para extracción de features o modelado posterior.

### 3.3 UMAP

Más rápido que t-SNE, mejor preservando la estructura global, y escalable a datasets más grandes. Cada vez más la opción por defecto para visualización de embeddings.

---

## 4. Consideraciones Prácticas

| Tarea | Algoritmo | Cuándo Usar |
|---|---|---|
| **Clustering** | K-Means | Datos grandes, clusters esféricos, $k$ conocido |
| | DBSCAN | Formas arbitrarias, $k$ desconocido, detección de outliers |
| | Jerárquico | Datos pequeños, querer dendrograma, interpretabilidad |
| **Red. dimensional** | PCA | Estructura lineal, velocidad, preprocesamiento antes de ML |
| | t-SNE | Solo visualización, datos pequeños a medianos |
| | UMAP | Visualización, datos grandes, preservación de estructura |
| **Anomalías** | Isolation Forest | Dimensiones altas, datos grandes |
| | LOF | Anomalías de densidad local |
| | Autoencoder | Patrones complejos, suficientes datos |

---

## 5. Errores Comunes

1. **PCA sin escalado**: si las features están en diferentes escalas, las direcciones de PCA están dominadas por features de alta varianza. Siempre estandarizar primero.

2. **Usar t-SNE para extracción de features**: t-SNE es no determinista y las distancias no son significativas. Es una herramienta de visualización, no un paso de preprocesamiento.

3. **Asumir que los clusters encontrados por K-Means son "reales"**: K-Means siempre encuentra $k$ clusters, incluso en datos aleatorios uniformes (ver [[Probability]]). Validar con score de silueta o conocimiento del dominio.

4. **No verificar outliers antes de K-Means**: los centroides pueden ser arrastrados por outliers. Eliminar o recortar valores extremos.

5. **Interpretar componentes de PCA como "features"**: cada componente es una combinación lineal de features originales. No son necesariamente interpretables.

---

## 6. Comprueba tu Conocimiento

1. Ejecutas K-Means con k=3 en datos aleatorios uniformes. ¿Cómo se verán los clusters? ¿Qué te dice esto sobre la evaluación de clustering?
2. ¿Por qué PCA requiere escalar los datos primero? ¿Qué pasa si lo omites?
3. DBSCAN etiqueta algunos puntos como "ruido" (-1). ¿Qué significa esto y por qué es útil?
4. Tienes 500 features y quieres visualizar tus datos. ¿Por qué PCA podría ser insuficiente y t-SNE una mejor opción?
5. ¿Cómo se relaciona la razón de varianza explicada con los eigenvalores en PCA?

---

## 7. Resumen

El aprendizaje no supervisado encuentra estructura oculta en datos sin etiquetas. Clustering agrupa puntos similares (K-Means para clusters esféricos, DBSCAN para formas arbitrarias). La reducción de dimensionalidad comprime datos preservando información (PCA para estructura lineal, t-SNE/UMAP para visualización). Estos métodos son esenciales para entender datos antes de aplicar aprendizaje supervisado, y para feature engineering en problemas de alta dimensionalidad.

---

## 8. ¿Dónde ir Siguente?

- [[Supervised Learning]] — Usar representaciones encontradas por métodos no supervisados
- [[Feature Engineering]] — PCA como técnica de extracción de features
- [[Model Evaluation]] — Evaluar la calidad de clustering
