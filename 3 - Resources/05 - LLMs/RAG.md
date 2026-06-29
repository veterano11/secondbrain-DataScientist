---
tags: [llm, rag, advanced]
status: growing
created: 2026-06-27
---

# Retrieval-Augmented Generation

## 1. Escenario de aprendizaje

Tu empresa quiere construir un chatbot que responda preguntas sobre su documentación interna de 10,000 páginas. El modelo base solo conoce información hasta su fecha de entrenamiento y no sabe nada de tus productos. Reentrenar el modelo sería prohibitivamente caro y lento. RAG resuelve esto: cada vez que un usuario hace una pregunta, buscas los documentos relevantes y se los pasas al modelo como contexto. El modelo ya no necesita saberlo todo — solo necesita leer lo que le das.

Los LLMs tienen una limitación fundamental: su conocimiento está congelado en el momento del entrenamiento, una consecuencia de cómo la [[Transformer Architecture]] almacena conocimiento en los pesos. No saben sobre eventos recientes, datos propietarios o documentos privados. Alucinan cuando se les pregunta sobre temas desconocidos.

RAG soluciona esto dando al modelo **acceso a conocimiento externo** en tiempo de inferencia. En lugar de depender únicamente de los parámetros del modelo, recuperamos documentos relevantes y los proporcionamos como contexto. Esto fundamenta la salida del modelo en información real, actual y verificable.

RAG es el enfoque estándar para aplicaciones empresariales de LLM — atención al cliente, preguntas y respuestas sobre documentos, asistentes de código y herramientas de investigación.

---

## 2. La Arquitectura RAG

```
Consulta → Embedder → Vector DB → Fragmentos Recuperados → LLM → Respuesta
```

### 2.1 Paso a Paso

1. **Indexación** (offline): dividir documentos en fragmentos → embedding de cada fragmento → almacenar en vector DB
2. **Consulta** (online): embedding de la pregunta del usuario → buscar en vector DB fragmentos similares → recuperar top-k
3. **Generación** (online): formatear prompt con fragmentos recuperados como contexto → el LLM genera la respuesta

### 2.2 La Clave

RAG NO actualiza los parámetros del LLM. Cambia solo la **ventana de contexto**. Esto significa:
- No se necesita entrenamiento con GPU
- Puede actualizar el conocimiento al instante (solo añade nuevos documentos al vector DB)
- El modelo base permanece sin cambios — sin riesgo de olvido catastrófico

---

## 3. Fragmentación (Chunking)

### 3.1 Por qué Importa la Fragmentación

Los documentos son demasiado largos para caber en una ventana de contexto. Incluso si pudieran, la recuperación funciona mejor en segmentos enfocados (fragmentos) que en documentos completos.

### 3.2 Estrategias

| Estrategia | Descripción | Mejor Para |
|---|---|---|
| **Tamaño fijo** | Dividir cada N caracteres/tokens con solapamiento | Simple, rápido |
| **Semántica** | Dividir en límites naturales (oraciones, párrafos) | Fragmentos coherentes |
| **Recursiva** | División jerárquica (documento → secciones → párrafos) | Docs estructurados |
| **Agentiva** | El LLM decide dónde dividir | Adaptativo, más lento |

### 3.3 Mejores Prácticas

- **Solapamiento**: 10-20% de solapamiento entre fragmentos asegura que no se pierda información en los límites
- **Tamaño**: 256-1024 tokens es el punto ideal — suficientemente grande para contener ideas completas, suficientemente pequeño para recuperación precisa
- **Metadatos**: almacenar fuente, título, número de página, sección con cada fragmento para citación y filtrado

---

## 4. Embeddings

### 4.1 Qué Hacen los Embeddings

Convertir texto en un vector denso (ej., 768 o 1536 dimensiones) donde la similitud semántica corresponde a la similitud vectorial, calculada por [[Neural Networks]].

$$\text{sim}(q, d) = \cos(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$

- "gato" y "gatito" → alta similitud coseno
- "gato" y "computadora" → baja similitud coseno

### 4.2 Modelos de Embedding Populares

| Modelo | Dimensiones | Mejor Para |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | 512-1536 | Propósito general |
| `text-embedding-3-large` (OpenAI) | 256-3072 | Alta precisión |
| `BGE-base` (BAAI) | 768 | Código abierto, buena calidad |
| `E5-mistral` (Microsoft) | 4096 | Alta precisión |
| `gte-large` (Alibaba) | 1024 | Código abierto |
| `nomic-embed-text` (Nomic) | 768 | Local, eficiente |

### 4.3 Mejores Prácticas de Embeddings

- **Normalizar embeddings**: asegura que la similitud coseno se comporte consistentemente
- **Multi-tarea**: algunos modelos se benefician de prefijar la consulta con "Representa esta oración para búsqueda: "
- **Reducción de dimensiones**: los modelos de embedding a menudo soportan reducir dimensiones (ej., 1536 → 256) con pérdida mínima de calidad

---

## 5. Recuperación

### 5.1 Recuperación Densa (Similitud de Embeddings)

Búsqueda por similitud vectorial. Entiende semántica — "coche" coincide con "vehículo" y "automóvil."

**Pros**: comprensión semántica
**Contras**: requiere un buen modelo de embedding, puede perder coincidencias exactas de palabras clave

### 5.2 Recuperación Dispersa (BM25)

Búsqueda por coincidencia de palabras clave. Las coincidencias exactas obtienen puntuaciones más altas.

**Pros**: no necesita entrenamiento, rápido, excelente para nombres propios y frases exactas
**Contras**: sin comprensión semántica

### 5.3 Recuperación Híbrida

Combinar ambas:

$$\text{puntaje} = \alpha \cdot \text{sim}_{\text{densa}} + (1-\alpha) \cdot \text{sim}_{\text{dispersa}}$$

Lo mejor de ambos mundos: comprensión semántica + coincidencia exacta. $\alpha$ se ajusta típicamente usando métodos de [[Statistics]] en un conjunto de validación (0.5-0.8).

### 5.4 Recuperación Avanzada

| Técnica | Descripción |
|---|---|
| **Multi-consulta** | Generar múltiples variaciones de la consulta, recuperar para cada una, deduplicar |
| **HyDE** | Generar un "documento hipotético" que responda la consulta, luego recuperar por su embedding |
| **RAPTOR** | Construir resúmenes jerárquicos del corpus, recuperar al nivel apropiado |
| **ColBERT** | Interacción tardía — coincidencia detallada a nivel de token |
| **Self-RAG** | Recuperar → generar → reflexionar sobre calidad → refinar |

---

## 6. Generación

### 6.1 La Estructura del Prompt

```
System: Eres un asistente útil. Responde la pregunta basándote SOLO en el contexto proporcionado.
Si el contexto no contiene suficiente información, di "No tengo suficiente información."

Contexto:
{fragmentos_recuperados}

Pregunta: {consulta}
Respuesta:
```

### 6.2 Ingeniería de Prompts para RAG

- **Enfatizar dependencia del contexto**: "Usa solo el contexto a continuación. No uses tu propio conocimiento."
- **Manejar información faltante**: "Si el contexto no responde la pregunta, di 'No encontrado en los documentos proporcionados.'"
- **Solicitar citas**: "Para cada afirmación, cita la fuente entre corchetes [fuente]."
- **Definir formato de salida**: JSON, viñetas o prosa según la aplicación.

---

## 7. Evaluación

| Métrica | Qué Mide |
|---|---|
| **Precisión de recuperación** | % de fragmentos recuperados que son relevantes |
| **Recall de recuperación** | % de fragmentos relevantes que fueron recuperados |
| **MRR** | Mean Reciprocal Rank — qué tan alto rankea el primer resultado relevante |
| **NDCG** | Normalized Discounted Cumulative Gain — calidad del ranking |
| **Relevancia de respuesta** | ¿La respuesta generada es útil para la consulta? |
| **Fidelidad** | ¿La respuesta se alinea con el contexto recuperado? |
| **Precisión del contexto** | ¿Todos los fragmentos de contexto son usados por el generador? |

**RAGAS** es un framework de evaluación dedicado para sistemas RAG, basado en principios generales de [[LLM Evaluation]] y proporcionando métricas automatizadas tanto para la calidad de recuperación como de generación.

---

## 8. Common Mistakes

1. **Fragmentos demasiado grandes**: un fragmento de 5000 tokens diluye la información relevante y desperdicia espacio en la ventana de contexto.

2. **No filtrar por metadatos**: recuperar de secciones irrelevantes o documentos desactualizados añade ruido. Filtra por fecha, fuente o categoría.

3. **Ignorar el orden de los fragmentos**: si la respuesta requiere combinar información de múltiples fragmentos, su orden en el contexto importa.

4. **Depender demasiado del conocimiento interno del LLM**: el modelo puede ignorar el contexto recuperado y responder desde sus propios datos de entrenamiento. El prompt debe priorizar explícitamente el contexto.

5. **Sin evaluación de la calidad de recuperación**: si la recuperación omite fragmentos relevantes, la generación no puede ser buena. Mide las métricas de recuperación por separado.

---

## 9. Check Your Understanding

1. Un usuario pregunta "¿Qué dijo el CEO sobre los resultados del Q4 en el informe anual de 2025?" pero tu base de conocimiento solo indexa hasta 2024. ¿Qué sucede? (No se recuperan fragmentos relevantes → el modelo dice "No encontrado.")

2. ¿Por qué importa el solapamiento de fragmentos? (La información en los límites puede dividirse entre fragmentos — el solapamiento asegura continuidad.)

3. Tu sistema RAG recupera 10 fragmentos pero solo 2 son relevantes. ¿Qué mejoras? (Recuperación — mejores embeddings, re-ranking o mejor fragmentación.)

4. BM25 coincide con palabras clave exactas. La recuperación densa coincide con significado semántico. ¿Cuándo funcionaría mejor BM25? (Nombres propios, códigos de producto, términos técnicos exactos.)

5. Un usuario hace una pregunta que requiere combinar información de 2 documentos diferentes. ¿Cómo maneja esto un sistema RAG estándar? (Ambos fragmentos se recuperan si son similares a la consulta; el LLM los combina.)

---

## 10. Resumen

RAG grounds LLM outputs in external knowledge by retrieving relevant chunks and feeding them as context. Good RAG requires: well-chunked documents, good embeddings, effective retrieval (hybrid is best), and careful prompt design that prioritizes context over the model's parametric knowledge. RAG enables up-to-date, verifiable, domain-specific AI applications without fine-tuning.

---

## 11. Where to Go Next

- [[Prompt Engineering]] — Diseñando prompts efectivos para RAG
- [[Fine-tuning]] — Un enfoque alternativo para adaptación de dominio
- [[Agentic Systems]] — RAG multi-paso con uso de herramientas
