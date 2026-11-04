"""
Day 14: RAG and Vector Databases

Demonstrates a complete RAG pipeline:
1. Document ingestion - Load learning notes as documents
2. Chunking - Split documents into manageable chunks
3. Embedding generation - Convert text to vector embeddings
4. Vector store - Store embeddings in searchable index
5. Retrieval - Find relevant chunks for a query
6. Response generation - Format results with citations

Integrates concepts from Days 1-13:
- Token counting (Day 1) - Track query/document sizes
- Context management (Day 2) - Handle retrieval token limits
- Prompt caching (Day 3) - Cache frequent document lookups
- Model comparison (Day 4) - Compare embedding approaches
- VRAM calculation (Day 8) - Estimate memory for vector store
- Structured outputs (Day 10) - Return results in JSON format
- Cost tracking (Days 3, 12) - Estimate RAG pipeline costs

Run: python experiments/14_rag_demo.py
"""

import json
import math
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# --- Data Models ---

@dataclass
class Document:
    """A document in the RAG system."""
    id: str
    title: str
    content: str
    source: str
    chunk_index: Optional[int] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result of a similarity search."""
    document: Document
    score: float  # Cosine similarity score
    relevance: str  # "high", "medium", "low"


@dataclass
class RagResponse:
    """Complete RAG response with citations."""
    query: str
    retrieved_chunks: List[RetrievalResult]
    answer: str
    sources: List[str]
    token_counts: Dict[str, int]
    estimated_cost: float
    processing_time_ms: float


# --- Simple Text Embedding ---

class SimpleEmbedding:
    """
    Simple bag-of-words embedding using TF-IDF-like approach.
    In production, use sentence-transformers or API embeddings.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace and punctuation."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return [t for t in tokens if len(t) > 2]  # Skip short tokens

    def _build_vocabulary(self, documents: List[str]):
        """Build vocabulary from all documents for IDF calculation."""
        from collections import defaultdict
        doc_freq = defaultdict(int)

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        self.doc_count = len(documents)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(
            sorted(doc_freq.items())[:1000]  # Limit vocabulary size
        )}

        # IDF: log(N / df)
        for word, df in doc_freq.items()[:1000]:
            self.idf[word] = math.log(self.doc_count / df) + 1

    def encode(self, text: str) -> np.ndarray:
        """Convert text to a fixed-dimension embedding vector."""
        if not self.vocabulary:
            # Fallback: simple hash-based embedding
            tokens = self._tokenize(text)
            if not tokens:
                return np.zeros(self.dimension)

            vec = np.zeros(self.dimension)
            for token in tokens:
                # Deterministic hash for consistent embeddings
                hash_val = hash(token) % self.dimension
                vec[hash_val] += 1.0

            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

        tokens = self._tokenize(text)
        vec = np.zeros(self.dimension)

        for token in tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                if idx < self.dimension:
                    vec[idx] += self.idf.get(token, 1.0)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Pad to dimension if needed
        if len(vec) < self.dimension:
            vec = np.pad(vec, (0, self.dimension - len(vec)))

        return vec[:self.dimension]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts (simulates batch embedding API)."""
        return np.array([self.encode(text) for text in texts])


# --- Vector Store ---

class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self, embedding_model: SimpleEmbedding, dimension: int = 256):
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        for doc in documents:
            doc.embedding = self.embedding_model.encode(doc.content)
            self.documents.append(doc)
            self.embeddings.append(doc.embedding)
        print(f"  Added {len(documents)} documents to vector store")

    def similarity_search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Find most similar documents using cosine similarity."""
        query_embedding = self.embedding_model.encode(query)

        if not self.embeddings:
            return []

        # Stack all embeddings into matrix
        doc_matrix = np.array(self.embeddings)

        # Cosine similarity: dot product / (norm product)
        # Our embeddings are already normalized, so just dot product
        similarities = np.dot(doc_matrix, query_embedding)

        # Get top-k indices
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            doc = self.documents[idx]

            # Determine relevance level
            if score > 0.3:
                relevance = "high"
            elif score > 0.15:
                relevance = "medium"
            else:
                relevance = "low"

            results.append(RetrievalResult(
                document=doc,
                score=score,
                relevance=relevance
            ))

        return results

    def __len__(self):
        return len(self.documents)


# --- Text Processor ---

class TextProcessor:
    """Handle document chunking and preprocessing."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Max words per chunk
            overlap: Words of overlap between chunks
        """
        # Simple sentence-based chunking
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_length + words > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap: keep last few sentences
                overlap_words = overlap // 10  # Rough overlap
                if overlap_words < len(current_chunk):
                    current_chunk = current_chunk[-overlap_words:]
                    current_length = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (from Day 1 concepts)."""
        return len(text) // 4


# --- RAG Pipeline ---

class RAGPipeline:
    """
    Complete RAG pipeline integrating all concepts from Days 1-14.

    Pipeline:
    1. Load documents
    2. Chunk documents
    3. Generate embeddings
    4. Store in vector database
    5. On query: retrieve relevant chunks
    6. Format response with citations
    """

    def __init__(self, chunk_size: int = 200, top_k: int = 3, dimension: int = 256):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embedding_model = SimpleEmbedding(dimension=dimension)
        self.vector_store = VectorStore(self.embedding_model, dimension)
        self.processor = TextProcessor()
        self.doc_cache: Dict[str, List[Document]] = {}  # Prompt caching (Day 3 concept)
        self.query_cache: Dict[str, List[RetrievalResult]] = {}

    def add_document(self, title: str, content: str, source: str = "") -> int:
        """
        Add a document to the RAG system.
        
        Args:
            title: Document title
            content: Full document content
            source: Source identifier
            
        Returns:
            Number of chunks created
        """
        cache_key = f"{title}_{len(content)}"
        if cache_key in self.doc_cache:
            # Use cached chunks (Day 3: prompt caching concept)
            for doc in self.doc_cache[cache_key]:
                self.vector_store.add_documents([doc])
            return len(self.doc_cache[cache_key])

        # Chunk the document
        chunks = self.processor.chunk_text(content, self.chunk_size)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=f"{title}_{i}",
                title=title,
                content=chunk,
                source=source,
                chunk_index=i
            )
            documents.append(doc)

        # Add to vector store
        self.vector_store.add_documents(documents)
        self.doc_cache[cache_key] = documents

        return len(chunks)

    def query(self, question: str, provider: str = "local") -> RagResponse:
        """
        Process a query using the RAG pipeline.
        
        Args:
            question: User's question
            provider: Which provider to estimate costs for
            
        Returns:
            Complete RAG response with citations
        """
        start_time = datetime.now()

        # Check cache first (Day 3: caching concept)
        if question in self.query_cache:
            cached_results = self.query_cache[question]
            return self._format_response(
                question, cached_results, start_time,
                from_cache=True
            )

        # Retrieve relevant chunks
        retrieved = self.vector_store.similarity_search(question, self.top_k)

        # Cache results
        self.query_cache[question] = retrieved

        return self._format_response(
            question, retrieved, start_time,
            from_cache=False,
            provider=provider
        )

    def _format_response(self, question: str, retrieved: List[RetrievalResult],
                         start_time: datetime, from_cache: bool = False,
                         provider: str = "local") -> RagResponse:
        """Format the RAG response with citations and cost estimation."""

        # Build context from retrieved chunks
        context_parts = []
        for result in retrieved:
            context_parts.append(
                f"[Source: {result.document.title}, Relevance: {result.relevance}]\n"
                f"{result.document.content.strip()}"
            )
        context = "\n\n".join(context_parts)

        # Generate answer (simulated - would be real LLM call in production)
        if retrieved:
            top_result = retrieved[0]
            answer = f"Based on your learning notes about {top_result.document.title}, "
            answer += f"the key points are:\n\n"

            # Extract key sentences from top result
            sentences = top_result.document.content.strip().split(". ")
            for sent in sentences[:5]:
                if len(sent) > 20:
                    answer += f"- {sent.strip()}...\n"

            answer += f"\n[Confidence: {top_result.score:.2f} | Relevance: {top_result.relevance}]"
        else:
            answer = "I couldn't find relevant information. Try rephrasing your question."

        # Token counting (Day 1 concept)
        input_tokens = (
            self.processor.estimate_tokens(question) +
            self.processor.estimate_tokens(context)
        )
        output_tokens = self.processor.estimate_tokens(answer)

        # Cost estimation (Days 3, 12 concepts)
        cost_per_1k = {"openai": 0.003, "anthropic": 0.003, "local": 0.0}
        cost = (input_tokens / 1000) * cost_per_1k.get(provider, 0.003)

        # Estimate VRAM for vector store (Day 8 concept)
        store_entries = len(self.vector_store)

        # Sources list
        sources = list(dict.fromkeys(  # Remove duplicates while preserving order
            r.document.title for r in retrieved
        ))

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return RagResponse(
            query=question,
            retrieved_chunks=retrieved,
            answer=answer,
            sources=sources,
            token_counts={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "context_chunks": len(retrieved),
                "vector_store_entries": store_entries,
            },
            estimated_cost=round(cost, 6),
            processing_time_ms=round(processing_time, 2)
        )


# --- Sample Documents ---

def get_sample_documents() -> List[Dict[str, str]]:
    """Sample documents - simplified versions of your learning notes."""
    return [
        {
            "title": "Day 1 - Tokenization",
            "content": """
            Tokenization is the process of converting text into numerical tokens that AI models can process.
            Each token represents roughly 4 characters of text in English, or about 3/4 of a word.
            The vocabulary size of a model determines how many unique tokens are available.
            Larger vocabularies provide better coverage of rare words but increase model size.
            Common tokenizers: BPE (Byte-Pair Encoding), WordPiece, SentencePiece.
            Tokenization is the first step in every LLM pipeline and affects all downstream costs.
            """,
            "source": "notes/day1_tokens.md"
        },
        {
            "title": "Day 6 - Transformers",
            "content": """
            Transformers use self-attention to process all tokens simultaneously.
            The attention mechanism creates weighted connections between all token pairs.
            Each token produces Query, Key, and Value vectors through learned linear projections.
            Multi-head attention allows the model to attend to information from different representation subspaces.
            The time complexity of self-attention is O(n^2) where n is sequence length.
            This quadratic cost is why long contexts require quadratic compute.
            """,
            "source": "notes/day6_transformers.md"
        },
        {
            "title": "Day 7 - Transformer Block",
            "content": """
            A transformer block has 6 steps: attention, residual connection, layer norm,
            feed-forward network, residual connection, layer norm.
            Residual connections prevent vanishing gradients in deep networks.
            Layer normalization stabilizes training by normalizing across features.
            Feed-forward networks add non-linear transformation capacity.
            Text generation is autoregressive: each token depends on all previous tokens.
            This sequential dependency is why generation is slower than processing input.
            """,
            "source": "notes/day7_transformer_block.md"
        },
        {
            "title": "Day 8 - VRAM and Quantization",
            "content": """
            VRAM is the dedicated memory on your GPU used to store model weights during inference.
            Larger models require more VRAM: 7B params in FP16 needs ~14GB, in INT4 ~3.5GB.
            Quantization reduces memory usage by lowering numerical precision.
            INT4 quantization (4-bit) provides 8x memory savings over FP32 with modest quality loss.
            KV cache stores attention keys and values, adding ~0.25GB per 2K context for INT4.
            CPU offloading moves layers to system RAM when GPU memory is insufficient.
            """,
            "source": "notes/day8_vram.md"
        },
        {
            "title": "Day 9 - Agent Patterns",
            "content": """
            An AI agent follows a loop: plan action, take action, observe result, replan.
            Agents use tools to extend capabilities beyond text generation.
            Key components: reasoning loop, tool registry, memory, planning.
            ReAct framework: Reason, Act, Observe, Repeat until goal achieved.
            Agents can make decisions, call APIs, search knowledge bases.
            Structured outputs ensure agents return reliable JSON instead of text.
            """,
            "source": "notes/day9_agents.md"
        },
        {
            "title": "Day 10 - API Patterns",
            "content": """
            LLM APIs use messages with roles: system, user, assistant.
            Structured outputs enforce response format using JSON schemas.
            Temperature controls randomness: 0.2 for facts, 0.7 for creativity.
            Retry logic with exponential backoff handles rate limits (HTTP 429).
            Cost is calculated per 1000 tokens, varies by provider.
            Provider abstraction allows switching between OpenAI and Anthropic.
            """,
            "source": "notes/day10_apis.md"
        },
        {
            "title": "Day 11 - Learning Assistant Agent",
            "content": """
            The Learning Assistant Agent integrates all previous concepts into a complete system.
            It uses conversation management, tool calling, and cost tracking.
            Key architecture: system prompt, user query, tool selection, response parsing.
            Context management trims long conversations to stay within token budgets.
            Cost tracking helps users understand pricing for each query.
            The agent learns from user interactions and maintains stateful sessions.
            """,
            "source": "notes/day11_agent.md"
        },
        {
            "title": "Day 12 - Provider Abstraction",
            "content": """
            The ProviderManager class allows switching between Anthropic, OpenAI, and Ollama.
            RetryWithRetry decorator implements exponential backoff: 1s, 2s, 4s delays.
            Fallback chains try primary provider then secondary if primary fails.
            Streaming response support provides token-by-token output for better UX.
            Cost tracking enables comparison: Anthropic $0.003/1K tokens, OpenAI $0.0025/1K.
            """
            + " Error handling distinguishes retryable errors (429, 500) from fatal errors (400, 401).",
            "source": "notes/day12_api_providers.md"
        },
        {
            "title": "Day 13 - Job Analyzer Agent",
            "content": """
            The Job Description Analyzer Agent parses job postings to extract required skills.
            It compares requirements against the user's current skill set to identify gaps.
            A weekly learning plan is generated, packing skill targets into 15-hour weeks.
            The agent recommends models compatible with the user's GTX 1650 hardware.
            """
            + " Project suggestions help build a portfolio for job applications.",
            "source": "notes/day13_job_analyzer.md"
        },
    ]


# --- Demo ---

QUESTIONS = [
    "What is tokenization?",
    "How does the transformer attention mechanism work?",
    "What quantization options exist for my GTX 1650?",
    "What are the key components of an AI agent?",
    "How can I reduce API costs when using LLMs?",
    "What tools does the Learning Assistant Agent use?",
    "Can you explain the difference between local and cloud LLMs?"
]


def demonstrate_rag():
    """Run the complete RAG pipeline demonstration."""
    print("=" * 70)
    print("DAY 14: RAG AND VECTOR DATABASES")
    print("=" * 70)
    print("Building a question-answering agent over your learning notes using RAG")
    print()

    # Create RAG pipeline
    pipeline = RAGPipeline(chunk_size=150, top_k=3, dimension=256)

    # Load sample documents
    print("Step 1: Loading documents...")
    docs = get_sample_documents()
    total_chunks = 0
    for doc in docs:
        chunks = pipeline.add_document(doc["title"], doc["content"], doc["source"])
        total_chunks += chunks
        print(f"  - {doc['title']}: {chunks} chunks")
    print(f"\nTotal documents: {len(docs)}, Total chunks: {total_chunks}")
    print(f"Vector store size: {len(pipeline.vector_store)} chunks")
    print(f"Vector dimension: {pipeline.vector_store.dimension}")

    # Demonstrate queries
    print("\n" + "=" * 70)
    print("Step 2: Query Processing with Vector Similarity Search")
    print("=" * 70)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {question}")
        print(f"{'='*70}")

        result = pipeline.query(question)

        print(f"\nRetrieved {len(result.retrieved_chunks)} relevant chunks:")
        for j, chunk in enumerate(result.retrieved_chunks, 1):
            print(f"\n  Chunk {j}: [{chunk.relevance.upper()}] {chunk.score:.3f}")
            print(f"  Source: {chunk.document.title} ({chunk.document.source})")
            preview = chunk.document.content.strip()[:120]
            print(f"  Preview: {preview}...")

        print(f"\nGenerated Answer:")
        print(f"{result.answer}")

        print(f"\nToken usage: {json.dumps(result.token_counts, indent=2)}")
        print(f"Estimated cost: ${result.estimated_cost:.6f}")
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Sources: {result.sources}")

    # Demonstrate caching
    print(f"\n{'='*70}")
    print("Step 3: Prompt Caching Demonstration")
    print(f"{'='*70}")
    print("Re-running the first query (should use cache)...")

    start = datetime.now()
    cached_result = pipeline.query(QUESTIONS[0])
    cache_time = (datetime.now() - start).total_seconds() * 1000

    print(f"Original time: {result.processing_time_ms}ms")
    print(f"Cached time: {cache_time:.2f}ms")
    print(f"Cache speedup: {result.processing_time_ms / max(cache_time, 1):.1f}x")
    print(f"Query cache entries: {len(pipeline.query_cache)}")

    # Memory estimation (Day 8 concept)
    print(f"\n{'='*70}")
    print("Step 4: Resource Estimation")
    print(f"{'='*70}")

    chunks = len(pipeline.vector_store)
    dim = pipeline.vector_store.dimension
    embedding_bytes = dim * 4  # float32 per dimension
    total_bytes = chunks * embedding_bytes

    print(f"Vector store:")
    print(f"  Documents: {chunks}")
    print(f"  Dimensions: {dim}")
    print(f"  Bytes per embedding: {embedding_bytes}")
    print(f"  Total memory: {total_bytes} bytes ({total_bytes / 1024:.1f} KB)")
    print(f"  Cost: $0.00 (in-memory numpy arrays)")

    # Compare with cloud alternatives
    print(f"\nCloud alternatives:")
    print(f"  Pinecone (serverless): ~$0.0001/query")
    print(f"  Weaviate: ~$0.01/1K objects stored")
    print(f"  ChromaDB (local): $0.00 (your electricity)")
    print(f"  FAISS (local): $0.00 (your electricity)")

    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("1. RAG combines retrieval (search) with generation (LLM) for grounded answers")
    print("2. Embeddings convert text to vectors where semantics = proximity")
    print("3. Chunking splits documents for better retrieval granularity")
    print("4. Cosine similarity finds the most relevant chunks for any query")
    print("5. Caching repeated queries saves compute and API costs")
    print("6. Local vector stores are free but cloud services offer persistence + scale")
    print("7. Token counting helps estimate context size before hitting limits")
    print(f"{'='*70}")


if __name__ == "__main__":
    demonstrate_rag()
