# Day 14: RAG and Vector Databases

## What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances an LLM by retrieving relevant documents before generating a response. Instead of relying solely on the model's training data, you ground responses in your own documents.

## Why RAG Matters

- **Fresh knowledge**: No retraining needed to incorporate new information
- **Reduced hallucinations**: Answers are grounded in retrieved documents
- **Domain specialization**: Customize knowledge without fine-tuning
- **Cost efficiency**: Use smaller/faster models for generation
- **Auditability**: You know exactly which documents influenced each answer

## Core Concepts

### Vector Embeddings

Text is converted to numerical vectors (embeddings) where semantic similarity = geometric proximity:

```
"Transformers use attention" → [0.2, -0.5, 0.8, ...]
"Attention mechanism connects tokens" → [0.3, -0.4, 0.7, ...]
                      ↑ These vectors are close in vector space
```

### Vector Database

Stores document embeddings and supports similarity search:
- **Local (FAISS, ChromaDB)**: Free, runs on your machine
- **Cloud (Pinecone, Weaviate, Qdrant)**: Managed, scalable, paid

### Cosine Similarity

Measures angle between vectors (ignores magnitude):
```python
similarity = dot_product(A, B) / (norm(A) * norm(B))
# Result: 1.0 (identical) to -1.0 (opposite), typical range 0.0-1.0
```

## RAG Pipeline

```
1. Document Ingestion
   ↓
2. Chunking (split into ~200 word chunks)
   ↓
3. Embedding Generation (text → vectors)
   ↓
4. Vector Store (store vectors)
   ↓
5. Query Processing
   ├── 5a. Embed query → vector
   ├── 5b. Find top-k most similar vectors
   └── 5c. Return matching documents
   ↓
6. Response Generation
   ├── 6a. Format prompt with retrieved documents
   ├── 6b. LLM generates answer
   └── 6c. Return with citations
```

## Chunking Strategies

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| Fixed-size (200 words) | Simple, fast | May split related content | General purpose |
| Sentence-based | Preserves meaning | Irregular sizes | FAQ, articles |
| Semantic | Thematically coherent | Computationally expensive | Long documents |
| Overlapping | Preserves context | More storage | Critical documents |

## Embedding Models

| Model | Dimensions | Size | Local? | Quality |
|-------|------------|------|--------|---------|
| MiniLM | 384 | 80MB | Yes | Good baseline |
| all-MiniLM-L6-v2 | 384 | 80MB | Yes | Good |
| text-embedding-ada-002 | 1536 | API | Yes | Excellent |
| text-embedding-3-small | 512 | API | Yes | Better |
| text-embedding-3-large | 3072 | API | Yes | Best |

For your GTX 1650: Use local embeddings (MiniLM) to avoid API costs.

## Vector Store Comparison

| Store | Local | Persistence | Scale | Cost |
|-------|-------|-------------|-------|------|
| FAISS | Yes | Manual | Limited | Free |
| ChromaDB | Yes | Yes | Medium | Free |
| Pinecone | No | Yes | Large | Paid |
| Weaviate | Yes/No | Yes | Large | Paid tier |

## Day 14 Experiment

The experiment (`experiments/14_rag_demo.py`) demonstrates a complete RAG pipeline:

1. **Document loading**: Your learning notes (Days 1-13) loaded as documents
2. **Chunking**: Documents split into 150-word overlapping chunks
3. **Embeddings**: TF-IDF-like bag-of-words vectors (dimension 256)
4. **Vector store**: In-memory NumPy vector store
5. **Query processing**: 7 sample questions tested
6. **Caching**: Repeated queries served from cache
7. **Cost tracking**: Estimated API costs per query

## Key Insights

1. **Context is king**: RAG's effectiveness depends on retrieval quality
2. **Chunking trade-off**: Small chunks = precise retrieval, large chunks = more context
3. **Top-k matters**: Too few → miss relevant docs; too many → noise
4. **Caching saves money**: Repeated queries don't need re-retrieval
5. **Local vs cloud**: Local is free but less scalable; cloud offers persistence

## Integration with Previous Days

| Concept | From Day | RAG Application |
|---------|----------|-----------------|
| Token counting | Day 1 | Estimate context size |
| Context management | Day 2 | Chunk size & overlap |
| Prompt caching | Day 3 | Cache query results |
| Model comparison | Day 4 | Embedding model selection |
| VRAM calculation | Day 8 | Vector store memory limits |
| Structured outputs | Day 10 | JSON response format |
| Provider abstraction | Day 12 | Choose retrieval backend |
| Agent loops | Day 9, 11 | RAG as an agent tool |

## Files Created

- `experiments/14_rag_demo.py` - Complete RAG pipeline implementation
- `notes/day14_rag.md` - This file

## Run

```bash
python experiments/14_rag_demo.py
```

## Next Steps

1. Replace simple embeddings with `sentence-transformers`
2. Add persistent storage (ChromaDB or FAISS index files)
3. Connect to a real LLM API for generation
4. Add hybrid search (keyword + semantic)
5. Implement RAG-as-a-tool for your Learning Assistant Agent
