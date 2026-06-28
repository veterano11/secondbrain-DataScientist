---
tags: [llm, rag, advanced]
status: growing
created: 2026-06-27
---

# Retrieval-Augmented Generation

## 1. Why This Matters

LLMs have a fundamental limitation: their knowledge is frozen at the time of training, a consequence of how the [[Transformer Architecture]] stores knowledge in weights. They do not know about recent events, proprietary data, or private documents. They hallucinate when asked about unfamiliar topics.

RAG solves this by giving the model **access to external knowledge** at inference time. Instead of relying solely on the model's parameters, we retrieve relevant documents and feed them as context. This grounds the model's output in real, current, verifiable information.

RAG is the standard approach for enterprise LLM applications — customer support, document Q&A, codebase assistants, and research tools.

---

## 2. The RAG Architecture

```
Query → Embedder → Vector DB → Retrieved Chunks → LLM → Answer
```

### 2.1 Step-by-Step

1. **Indexing** (offline): split documents into chunks → embed each chunk → store in vector DB
2. **Querying** (online): embed the user's question → search vector DB for similar chunks → retrieve top-k
3. **Generation** (online): format prompt with retrieved chunks as context → LLM generates answer

### 2.2 The Key Insight

RAG does NOT update the LLM's parameters. It changes only the **context window**. This means:
- No GPU training needed
- Can update knowledge instantly (just add new documents to the vector DB)
- The base model stays unchanged — no risk of catastrophic forgetting

---

## 3. Chunking

### 3.1 Why Chunking Matters

Documents are too long to fit in a context window. Even if they could, retrieval works better on focused segments (chunks) than on entire documents.

### 3.2 Strategies

| Strategy | Description | Best For |
|---|---|---|
| **Fixed size** | Split every N characters/tokens with overlap | Simple, fast |
| **Semantic** | Split at natural boundaries (sentences, paragraphs) | Coherent chunks |
| **Recursive** | Hierarchical splitting (document → sections → paragraphs) | Structured docs |
| **Agentic** | LLM decides where to split | Adaptive, slower |

### 3.3 Best Practices

- **Overlap**: 10-20% overlap between chunks ensures no information is lost at boundaries
- **Size**: 256-1024 tokens is the sweet spot — large enough to contain complete ideas, small enough for precise retrieval
- **Metadata**: store source, title, page number, section with each chunk for citation and filtering

---

## 4. Embeddings

### 4.1 What Embeddings Do

Convert text into a dense vector (e.g., 768 or 1536 dimensions) where semantic similarity corresponds to vector similarity, computed by [[Neural Networks]].

$$\text{sim}(q, d) = \cos(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$

- "cat" and "kitten" → high cosine similarity
- "cat" and "computer" → low cosine similarity

### 4.2 Popular Embedding Models

| Model | Dimensions | Best For |
|---|---|---|
| `text-embedding-3-small` (OpenAI) | 512-1536 | General purpose |
| `text-embedding-3-large` (OpenAI) | 256-3072 | High accuracy |
| `BGE-base` (BAAI) | 768 | Open source, good quality |
| `E5-mistral` (Microsoft) | 4096 | High accuracy |
| `gte-large` (Alibaba) | 1024 | Open source |
| `nomic-embed-text` (Nomic) | 768 | Local, efficient |

### 4.3 Embedding Best Practices

- **Normalize embeddings**: ensures cosine similarity behaves consistently
- **Multi-task**: some models benefit from prefixing the query with "Represent this sentence for search: "
- **Dimension reduction**: embedding models often support reducing dimensions (e.g., 1536 → 256) with minimal quality loss

---

## 5. Retrieval

### 5.1 Dense Retrieval (Embedding Similarity)

Search by vector similarity. Understands semantics — "car" matches "vehicle" and "automobile."

**Pros**: semantic understanding
**Cons**: requires good embedding model, can miss exact keyword matches

### 5.2 Sparse Retrieval (BM25)

Search by keyword overlap. Exact word matches get higher scores.

**Pros**: no training needed, fast, great for proper nouns and exact phrases
**Cons**: no semantic understanding

### 5.3 Hybrid Retrieval

Combine both:

$$\text{score} = \alpha \cdot \text{sim}_{\text{dense}} + (1-\alpha) \cdot \text{sim}_{\text{sparse}}$$

Best of both worlds: semantic understanding + exact match. $\alpha$ is typically tuned using [[Statistics]] methods on a validation set (0.5-0.8).

### 5.4 Advanced Retrieval

| Technique | Description |
|---|---|
| **Multi-query** | Generate multiple query variations, retrieve for each, deduplicate |
| **HyDE** | Generate a "hypothetical document" that answers the query, then retrieve by its embedding |
| **RAPTOR** | Build hierarchical summaries of the corpus, retrieve at appropriate level |
| **ColBERT** | Late interaction — fine-grained token-level matching |
| **Self-RAG** | Retrieve → generate → reflect on quality → refine |

---

## 6. Generation

### 6.1 The Prompt Structure

```
System: You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context does not contain enough information, say "I don't have enough information."

Context:
{retrieved_chunks}

Question: {query}
Answer:
```

### 6.2 Prompt Engineering for RAG

- **Emphasize context reliance**: "Only use the context below. Do not use your own knowledge."
- **Handle missing information**: "If the context does not answer the question, say 'Not found in the provided documents.'"
- **Request citations**: "For each claim, cite the source in brackets [source]."
- **Define output format**: JSON, bullet points, or prose depending on the application.

---

## 7. Evaluation

| Metric | What It Measures |
|---|---|
| **Retrieval precision** | % of retrieved chunks that are relevant |
| **Retrieval recall** | % of relevant chunks that were retrieved |
| **MRR** | Mean Reciprocal Rank — how high the first relevant result ranks |
| **NDCG** | Normalized Discounted Cumulative Gain — ranking quality |
| **Answer relevance** | Is the generated answer useful for the query? |
| **Faithfulness** | Does the answer align with the retrieved context? |
| **Context precision** | Are all context pieces used by the generator? |

**RAGAS** is a dedicated evaluation framework for RAG systems, building on general [[LLM Evaluation]] principles and providing automated metrics for both retrieval and generation quality.

---

## 8. Common Mistakes

1. **Chunks that are too large**: a 5000-token chunk dilutes relevant information and wastes context window space.

2. **Not filtering by metadata**: retrieving from irrelevant sections or outdated documents adds noise. Filter by date, source, or category.

3. **Ignoring chunk order**: if the answer requires combining information from multiple chunks, their order in the context matters.

4. **Over-relying on the LLM's internal knowledge**: the model may ignore retrieved context and answer from its own training data. The prompt must explicitly prioritize context.

5. **No evaluation of retrieval quality**: if retrieval misses relevant chunks, generation cannot be good. Measure retrieval metrics separately.

---

## 9. Check Your Understanding

1. A user asks "What did the CEO say about Q4 results in the 2025 annual report?" but your knowledge base only indexes through 2024. What happens? (No relevant chunks retrieved → model says "Not found.")

2. Why does chunk overlap matter? (Information at boundaries can be split across chunks — overlap ensures continuity.)

3. Your RAG system retrieves 10 chunks but only 2 are relevant. What do you improve? (Retrieval — better embeddings, reranking, or better chunking.)

4. BM25 matches exact keywords. Dense retrieval matches semantic meaning. When would BM25 perform better? (Proper nouns, product codes, exact technical terms.)

5. A user asks a question that requires combining information from 2 different documents. How does a standard RAG system handle this? (Both chunks are retrieved if similar to the query; the LLM combines them.)

---

## 10. Summary

RAG grounds LLM outputs in external knowledge by retrieving relevant chunks and feeding them as context. Good RAG requires: well-chunked documents, good embeddings, effective retrieval (hybrid is best), and careful prompt design that prioritizes context over the model's parametric knowledge. RAG enables up-to-date, verifiable, domain-specific AI applications without fine-tuning.

---

## 11. Where to Go Next

- [[Prompt Engineering]] — Designing effective RAG prompts
- [[Fine-tuning]] — An alternative approach for domain adaptation
- [[Agentic Systems]] — Multi-step RAG with tool use
