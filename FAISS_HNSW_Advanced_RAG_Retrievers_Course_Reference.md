# FAISS, HNSW & Advanced RAG Retrievers — Course Reference

*A consolidated study reference covering FAISS vs. Chroma DB vs. Milvus, FAISS index types, a deep dive into the HNSW algorithm (construction and search), and advanced retrieval strategies in LlamaIndex and LangChain.*

> This is a companion reference to the earlier **"GenAI, LangChain & Flask"**, **"RAG, LlamaIndex & Gradio"**, and **"Vector Databases, Chroma DB & Similarity Search"** files. Chroma DB's own HNSW configuration and basic similarity search are covered there; this file goes deeper into HNSW's internals and surveys the broader vector-search and retriever landscape.

---

## Table of Contents

1. [FAISS vs. Chroma DB vs. Milvus](#1-faiss-vs-chroma-db-vs-milvus)
2. [FAISS Index Types](#2-faiss-index-types)
3. [HNSW Deep Dive](#3-hnsw-deep-dive)
4. [Advanced Retrievers for RAG — Core Concepts](#4-advanced-retrievers-for-rag--core-concepts)
5. [LlamaIndex — Index Types & Retrievers](#5-llamaindex--index-types--retrievers)
6. [LangChain Retrievers](#6-langchain-retrievers)
7. [Decision Framework](#7-decision-framework)
8. [Cheat Sheets Recap](#8-cheat-sheets-recap)

---

## 1. FAISS vs. Chroma DB vs. Milvus

### 1.1 What Is FAISS?

**FAISS** (Facebook AI Similarity Search) is a **library** made by Meta for fast vector search. It runs on a **single machine** (CPU or GPU), has **no built-in database or server** — you use it by writing code — and is ideal when you want **full control and high performance**.

### 1.2 What Is Chroma DB (Recap)?

**Chroma DB** is a **full vector database** built for AI use cases. It stores both vectors **and** metadata (tags, descriptions), can run locally or as a server, and integrates easily with tools like LangChain.

### 1.3 Side-by-Side Comparison

| Aspect | FAISS | Chroma DB |
|---|---|---|
| **Type** | Library | Full database |
| **Deployment** | Single-node only, no native distributed scaling | Single-node **and** distributed deployments |
| **Usage** | Code-based integration, no server component | Can run locally or as a server |
| **Control** | Full control over indexing and performance | Less low-level control |
| **Metadata** | No native metadata support | Native support for storing and filtering metadata |
| **Indexing options** | Many (Flat, IVF, LSH, HNSW, etc.) | Only **HNSW** |
| **Integration** | Works with LangChain and LlamaIndex | Works with LangChain and LlamaIndex |

### 1.4 Extending FAISS with Milvus

FAISS is powerful for **local, high-performance** vector search, but lacks metadata support and distributed scaling. **Milvus** is a vector database that uses **FAISS as one of its core indexing engines** and adds the missing capabilities:
- **Metadata support** — storing and filtering metadata alongside vectors, enabling hybrid queries (e.g., "find similar items under $50").
- **Distributed deployments** — suitable for large-scale production environments.
- **Scalability** — addresses FAISS's single-node limitation, while retaining FAISS-level performance.

### 1.5 When to Use Each

| Use... | When... |
|---|---|
| **FAISS** | You want full control and performance on a single machine; you need access to multiple indexing algorithms; you're building custom, high-performance applications; metadata support isn't required (or can be handled externally) |
| **Chroma DB** | You need quick AI development/prototyping; metadata-rich queries matter; you want easy integration with AI tools; you need both single-node and distributed options |
| **Milvus** | You need a scalable, production-ready vector database; hybrid search is desired; distributed capabilities are essential; you want FAISS-level performance plus database features |

**Bottom line**: both FAISS and Chroma DB work with LangChain and LlamaIndex for RAG pipelines — choose based on **project size, complexity, and infrastructure requirements**.

---

## 2. FAISS Index Types

Each FAISS index type balances **speed, memory, and accuracy** differently.

### 2.1 Flat Index

Compares the distance (Euclidean or dot product) between the query embedding and **every** vector in the store via **brute-force search**, then retrieves the *k* nearest vectors, ranked closest→farthest.

- **Method**: brute-force comparison with all vectors.
- **Accuracy**: very accurate.
- **Performance**: very slow for large datasets.
- **Use case**: small datasets where accuracy is critical.

### 2.2 Inverted File Index (IVF)

Speeds up search by **clustering** vectors (e.g., via k-means) into **Voronoi cells** around centroids; each cell holds the vectors closest to its centroid. A query is compared only against the nearest cell(s), reducing computation.

- **Clustering**: k-means → Voronoi cells.
- **Search strategy**: query searches only the nearest cells.
- **Trade-off**: faster than Flat, but may slightly reduce accuracy.
- **Limitation**: some genuinely nearby vectors might live in a different cell and get missed.
- **Best for**: medium-to-large datasets (though HNSW may outperform it in many cases).

### 2.3 Locality-Sensitive Hashing (LSH)

Uses hash functions that map **similar vectors to the same bucket**, enabling fast, memory-efficient search; a query searches only the closest matching bucket(s).

- **Method**: hash functions group similar vectors into buckets.
- **Performance**: fast and memory-efficient.
- **Best use**: high-dimensional sparse data, such as text embeddings.
- **Trade-off**: neither the fastest nor the most accurate method; less commonly used today.

### 2.4 Hierarchical Navigable Small World (HNSW) — Summary

Organizes vectors into a **hierarchy of layers**: sparse top layers act as "express highways" to quickly approach the target region, while lower layers are progressively denser with detailed local connections. Search starts at the top and moves downward, using the best candidate from each layer as the entry point into the next.

- **Performance**: both fast and accurate, especially for large datasets.
- **Complexity**: the hierarchical structure enables **O(log n)** search complexity (vs. brute-force's O(n)).
- **Recall**: typically **90–99%**.
- *(See Section 3 for the full deep dive — navigation analogy, construction process, and tunable parameters.)*

### 2.5 FAISS Index Selection Summary

| Index | Accuracy | Speed | Best For |
|---|---|---|---|
| **Flat** | Highest | Slowest | Small datasets where accuracy is critical |
| **IVF** | Balanced | Balanced | Medium-to-large datasets (may be outperformed by HNSW) |
| **LSH** | Lower | Fast, memory-efficient | High-dimensional sparse data (less commonly used) |
| **HNSW** | High | Fast | Medium-to-large datasets — strong performance & scalability |

---

## 3. HNSW Deep Dive

### 3.1 The Big Picture Analogy

Finding a similar item among millions is like finding a specific restaurant in a huge city: instead of checking every restaurant one by one, you use a navigation system that starts with a zoomed-out view of major highways, then gradually zooms in to smaller streets, arriving at the exact destination. HNSW applies this idea to **high-dimensional data** — the meaning of sentences, image characteristics, patterns in music, etc.

HNSW was introduced in a 2016 paper by **Yu. A. Malkov and D. A. Yashunin**, *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"* (arXiv:1603.09320; later published in IEEE TPAMI) — one of the most cited works in similarity search.

### 3.2 Building Block 1 — Small World Networks

Research shows any two people in the world are connected by roughly **six degrees of separation** — a "small world." Small-world networks have two key properties:
- **High clustering coefficient** — nodes form tight-knit groups with many connections between neighbors.
- **Low average path length** — despite the clustering, any node can reach any other node in just a few steps.

HNSW applies this to data: data points connect to **similar** data points, forming a network you can traverse quickly from any point to any other. This concept was formalized by **Duncan Watts and Steven Strogatz** (1998) and underlies many modern search algorithms.

### 3.3 Building Block 2 — Navigable Networks & Greedy Routing

A **navigable** network has smart (not random) connections that help you move toward your target — like road signs. The **Navigable Small World (NSW)** algorithm, HNSW's foundation, uses **greedy routing**:
1. Start at an entry point.
2. Look at all directly connected neighbors.
3. Move to whichever neighbor is closest to the target.
4. Repeat until you can't get any closer.

This greedy search has **polylogarithmic time complexity O(logᵏ n)** — far faster than a naive linear scan.

### 3.4 Building Block 3 — Hierarchical Structure

Instead of one flat network, HNSW builds **multiple layers**, like a skyscraper — inspired by **skip lists** (a probabilistic data structure maintaining multiple levels of linked lists, invented by William Pugh in 1989).

| Layer | Analogy | Characteristics |
|---|---|---|
| **Top layer(s)** | Express highways | Very few data points; long-distance links that jump far across the data space. Only about `1/2^L` nodes appear at layer `L` — probability decreases exponentially with height |
| **Middle layers** | Main roads | More points, medium-distance connections; progressively shorter-range connections as you go down |
| **Ground floor (Layer 0)** | Local streets | **Every** data point in the dataset exists here, connected by short-distance links |

Each node's layer assignment is chosen **probabilistically** via an exponentially decaying distribution — most points only reach Layer 0; a few reach higher layers; very few reach the top.

### 3.5 How an HNSW Search Works — Step by Step

Using a worked example with 12 data points (green circles) and a query (red square):

1. **Start with the HNSW index.** Some data points (including conceptually the query) are projected onto higher layers; identical points across layers are linked by dotted vertical lines. Connections in higher layers do **not** necessarily mirror those in lower layers.
2. **Enter at a random point in the top layer** (e.g., point 1 in Layer 2).
3. **Perform greedy search in the current layer**:
   - Compute the distance between the entry point and the query.
   - Compute distances to all of the entry point's neighbors in this layer.
   - Identify the closest point.
   - If the closest point *is* the entry point, or has no unexplored neighbors → move to the next layer.
   - Otherwise, set the closest point as the new entry point and repeat.
4. **Move down one layer and repeat**, reusing already-computed distances where possible and only computing distances to *new* neighbors. The best point found becomes the entry point for the next (lower) layer.
5. **Final search in the bottom layer (Layer 0)** — continue greedy search among all neighbors until no closer point remains.
6. **Return the approximate nearest neighbor** — the algorithm can return multiple approximate nearest neighbors if requested.

**Efficiency in the worked example**: the brute-force approach would require 12 distance computations (one per data point); the HNSW walk-through needed only **8** — and the efficiency gain grows much larger on bigger, real-world datasets.

> **Note**: HNSW does **not** guarantee the exact nearest neighbor (due to its greedy nature) but typically finds a very close match.

### 3.6 How the HNSW Index Is Built — Step by Step

1. **Start with an empty graph.** The first inserted point becomes the entry point for all future insertions/searches.
2. **Assign a height (layer) to each new data point** — chosen randomly via an exponentially decreasing probability distribution (most points land only on Layer 0; progressively fewer reach each higher layer).
3. **Insert the point into the graph**:
   - **a. Start at the top layer** where the current entry point exists; use greedy search (look at neighbors, move to whichever is closest to the *new* point) until no closer neighbor exists in that layer.
   - **b. Move down one layer**, repeating greedy search from the best point found above.
   - **c. Connect to neighbors** — at each layer from the top down to Layer 0, the new point connects to its **M** closest neighbors; connections are **bidirectional**.
4. **Repeat for every new point**, gradually building a multi-layered graph that is fast to search.

### 3.7 Key Tunable Parameters

| Parameter | Role | Trade-off |
|---|---|---|
| **M** (max connections per node) | How many neighbors each point connects to | Higher M → better accuracy/recall, more memory usage. Lower M → faster build, less memory, lower accuracy |
| **`efConstruction`** (search breadth during build) | Candidates considered when finding neighbors during insertion | Higher → better graph quality, slower build. Lower → faster build, possibly worse search quality later |
| **`efSearch`** (search breadth during querying) | Candidate nodes explored per query | Higher → better accuracy, slower search. Lower → faster search, may miss best matches. **This is the main knob for tuning speed vs. accuracy at query time** |
| **`ml`** (level multiplier) | Affects how likely a point is to land in a higher layer | Controls the overall shape/height of the hierarchy |

**Tuning guidelines**: start with defaults (**M=16, efConstruction=200**); increase M for higher recall (up to ~M=64); adjust `efSearch` based on your speed/accuracy needs; use benchmarking tools to find the optimum for your data.

### 3.8 Why This Works

- **Upper layers** provide "highway" connections for long-distance jumps across the data space.
- **The bottom layer** gives fine-grained, local detail.
- Together, this combination makes HNSW both **fast and accurate**, even on huge datasets.
- **Scale separation + polylogarithmic complexity** (`O(logᵏ n)`) + **high-probability guarantees** (mathematical proofs show a high probability of near-optimal results) are the theoretical properties underpinning HNSW's performance.

### 3.9 Limitations & Trade-offs

| Limitation | Detail |
|---|---|
| **Approximate results** | Typical recall is 90–99% — it may occasionally miss the true nearest neighbor; usually an acceptable trade-off |
| **Parameter tuning required** | Getting the best performance means tuning M, `efConstruction`, `efSearch`, and `ml` |
| **Dynamic updates** | Best suited to **mostly-static** datasets; frequent insertions/deletions degrade the index over time and may require **periodic reconstruction** |
| **Distance metric limitations** | Works best with **Euclidean (L2)** and **cosine similarity**; other metrics may need modification or underperform |

### 3.10 When to Use (and Not Use) HNSW

**Use HNSW when you need:**
- Fast similarity search in large datasets.
- Good accuracy with reasonable speed.
- To handle high-dimensional data (many features).
- Scalable solutions that keep working as data grows.

**Consider alternatives when:**
- You need **guaranteed exact** results.
- Your dataset is **very small** (simpler methods may suffice).
- **Memory usage** is a critical constraint.
- You need to **frequently add/remove data** (HNSW favors mostly-static data).

### 3.11 The Science Behind HNSW

HNSW's breakthrough combined two existing ideas:
1. **Skip lists** (William Pugh, 1989) — a probabilistic structure enabling fast search via multiple levels of linked lists.
2. **Navigable small-world networks** — reaching any point quickly via smart connections, building on **Kleinberg's** work on navigable networks.

Published by Malkov & Yashunin in 2016, the paper has been cited **2,000+ times** and is one of the most influential works in similarity search.

---

## 4. Advanced Retrievers for RAG — Core Concepts

### 4.1 What Are Advanced Retrievers?

Advanced retrievers go beyond simple vector similarity search to provide more nuanced, context-aware retrieval through:
- **Semantic understanding** — embeddings for meaning/context.
- **Keyword matching** — precise term-based search for exact specifications.
- **Hierarchical context** — maintaining relationships between levels of information (e.g., parent/child chunks).
- **Multi-query processing** — generating and combining results from multiple query variations.
- **Fusion techniques** — intelligently combining results from different retrieval methods.

### 4.2 Maximum Marginal Relevance (MMR)

- **Purpose**: balance **relevance** and **diversity** in retrieved results.
- **Method**: select documents that are highly relevant to the query **and** minimally similar to documents already selected.
- **Benefit**: avoids redundancy and ensures comprehensive coverage of different aspects of the query.

---

## 5. LlamaIndex — Index Types & Retrievers

### 5.1 Core Index Types

| Index Type | Function | Best Suited For |
|---|---|---|
| **`VectorStoreIndex`** | Stores vector embeddings for each document chunk | Semantic retrieval based on meaning; commonly used in LLM/RAG pipelines |
| **`DocumentSummaryIndex`** | Generates and stores document **summaries** at indexing time, used to filter documents before retrieving full content | Large/diverse document sets that can't fit in an LLM's or embedding model's context window; returns the **original documents**, not the summaries |
| **`KeywordTableIndex`** | Extracts keywords from documents and maps them to content chunks | Exact keyword matching; rule-based or hybrid search |

### 5.2 LlamaIndex Retriever Types

#### 1. Vector Index Retriever
The **most common** retriever — embeds the query and compares it to document embeddings using cosine similarity.
- **Ideal for**: general-purpose search, RAG pipelines where semantic understanding matters.
- **Limitation**: may miss exact keyword matches when specific terms are crucial.

#### 2. BM25 Retriever (Keyword-Based)
An **advanced keyword-based** method that improves on **TF-IDF**.

**TF-IDF foundation**:
- **Term Frequency (TF)** — how often a word appears in a document.
- **Inverse Document Frequency (IDF)** — how rare that word is across all documents.
- **TF-IDF score** = TF × IDF — highlights words frequent in one document but rare across the collection.

**BM25 improvements over TF-IDF**:
- **Term frequency saturation** — reduces the impact of repeated terms via a saturation function.
- **Document length normalization** — adjusts for document length, preventing a bias toward longer documents.
- **Tunable parameters**: `k1 ≈ 1.2` (saturation control), `b ≈ 0.75` (length normalization).
- **Best for**: technical documentation, legal documents, exact-terminology requirements.

#### 3. Document Summary Index Retrievers
Use document **summaries** (rather than the actual document text) to find relevant content, then return the **original documents**. Two variants:
- **`DocumentSummaryIndexLLMRetriever`** — uses an LLM to analyze the query against summaries. Intelligent, but more time-consuming and expensive.
- **`DocumentSummaryIndexEmbeddingRetriever`** — uses semantic similarity between the query and summary embeddings. Faster and more cost-effective; better for large collections.

#### 4. Auto Merging Retriever
Preserves context in long documents using a **hierarchical structure** (parent/child nodes via hierarchical chunking).
- If **enough child nodes from the same parent** are retrieved, the retriever returns the **parent node** instead — consolidating related content and preserving broader context.
- **Dual storage**: child chunks for precise matching, parent chunks for context.
- **Best for**: long documents, legal papers, technical specifications.

#### 5. Recursive Retriever
Follows **relationships between nodes** via references — e.g., citations in an academic paper, or other metadata links.
- Supports both **chunk references** and **metadata references**.
- **Best for**: academic papers with citations, interconnected knowledge bases.

#### 6. Query Fusion Retriever
Combines results from **different retrievers** (e.g., vector-based + keyword-based) and can optionally generate **multiple query variations** via an LLM to improve coverage. Results are merged using a **fusion strategy**:

| Fusion Strategy | How It Works | Best For |
|---|---|---|
| **Reciprocal Rank Fusion (RRF)** | Combines ranked lists by giving higher scores to documents that appear near the top of *any* list. Formula: `RRF_score(d) = Σ (1 / (rank_i(d) + k))`, with `k ≈ 60`. Robust; doesn't rely on score magnitudes | Default choice for most fusion scenarios, production systems |
| **Relative Score Fusion** | Normalizes scores within each result set by dividing by the max score (`normalized_score = original_score / max_score`), preserving each retriever's relative confidence | When embedding-model confidence scores are meaningful |
| **Distribution-Based Score Fusion** | Most sophisticated — uses statistical techniques (z-score normalization, percentile ranking) to combine results, handling score variability | Complex queries with varying score distributions |

### 5.3 LlamaIndex Retriever Recommendations by Use Case

| Use Case | Recommended Retriever(s) |
|---|---|
| **General Q&A** | Vector Index Retriever, potentially combined with BM25 (semantic + keyword) |
| **Technical documents** (exact terms matter) | BM25 as primary, Vector Index Retriever as secondary for contextual flexibility |
| **Long documents** | Auto Merging Retriever (returns parent only when enough children are retrieved) |
| **Research papers** | Recursive Retriever (to pull in cited-paper content) |
| **Large document sets** | Document Summary Index Retriever to narrow candidates, then Vector Search within the remaining subset |

---

## 6. LangChain Retrievers

### 6.1 The Retriever Interface

LangChain defines a **retriever** as *"an interface that returns documents based on an unstructured query."* It's **more general than a vector store**: it accepts a string query and returns a list of documents, but doesn't necessarily store documents itself — its purpose is purely to **retrieve** them.

### 6.2 Vector Store-Backed Retriever
The **foundation retriever** — a lightweight wrapper around a vector store. Supports three search types:
- **Simple similarity search** — returns documents ranked by similarity (default: 4 results).
- **MMR search** — balances relevance and diversity to avoid redundancy (see Section 4.2).
- **Similarity score threshold** — returns only documents above a specified similarity threshold.

### 6.3 Multi-Query Retriever
**Problem addressed**: distance-based vector retrieval can vary with subtle changes in query wording.

**Process**:
1. Uses an LLM to generate **multiple queries** from different perspectives.
2. Retrieves a set of relevant documents for **each** query.
3. Takes the **unique union** of all results for a broader candidate set.

**Benefit**: generating multiple perspectives on the same question can overcome some limitations of pure distance-based retrieval.

### 6.4 Self-Querying Retriever
**Core capability**: it can "query itself" — converting a natural-language query into a **structured query** with two parts:
1. A **string to look up semantically**.
2. A **metadata filter** to accompany it.

**Requirements**: documents need rich, structured metadata with field descriptions.
**Best for**: combining semantic search with attribute filtering.

**Example queries**:
- *"I want to watch a movie rated higher than 8.5"* → filter only.
- *"Has Greta Gerwig directed any movies about women"* → semantic query + filter.

### 6.5 Parent Document Retriever
**Problem solved**: the "conflicting desires" of chunking — small chunks give accurate embeddings, but large chunks preserve context.

**Solution**: split and store **small** chunks for retrieval accuracy, but return the **larger parent document** for context.

**Process**:
1. During retrieval, first fetch the small chunks.
2. Look up the **parent IDs** for those chunks.
3. Return the larger documents containing those chunks.

**Architecture**:
- **Two splitters** — a parent splitter (large chunks for context) and a child splitter (small chunks for embeddings).
- **Dual storage** — a vector store for the child-chunk embeddings, and a document store for the parent documents.

---

## 7. Decision Framework

| Need | LlamaIndex Choice | LangChain Choice |
|---|---|---|
| **Exact keyword matching** | BM25 Retriever | Vector Store-Backed + custom keyword logic |
| **Multi-query with fusion** | Query Fusion Retriever (RRF / Relative / Distribution-based) | Multi-Query Retriever (union approach) |
| **Citation following** | Recursive Retriever | Not directly supported |
| **Hierarchical context** | Auto Merging Retriever | Parent Document Retriever |
| **Simple semantic search** | Vector Index Retriever | Vector Store-Backed Retriever |

---

## 8. Cheat Sheets Recap

### 8.1 Cheat Sheet: Build a Comprehensive RAG Application (FAISS vs. Chroma DB, HNSW)

- Full FAISS vs. Chroma DB technology comparison (Section 1.3).
- FAISS index types — Flat, IVF, LSH, HNSW — with characteristics and use cases (Section 2).
- HNSW architecture summary: top layers as "express highways," lower layers denser, greedy top-down search.
- HNSW key parameters: `M`, `efConstruction`, `efSearch`, `ml` (Section 3.7).
- HNSW limitations: approximate results (90–99% recall), poor fit for dynamic/frequently-updated data, best with L2/cosine distance.
- Extending FAISS with Milvus for metadata + distributed scaling (Section 1.4).
- When to use FAISS vs. Chroma DB vs. Milvus (Section 1.5).

### 8.2 Reading: Hierarchical Navigable Small World (HNSW)

- Full conceptual building blocks: small-world networks, navigable networks/greedy routing, hierarchical structure/skip lists (Sections 3.2–3.4).
- Step-by-step worked search example across 3 layers with 12 data points (Section 3.5).
- Step-by-step index-construction process: empty graph → layer assignment → greedy insertion → bidirectional neighbor connections (Section 3.6).
- Full parameter-tuning guidance, including starting defaults `M=16, efConstruction=200` (Section 3.7).
- The historical/scientific background: Malkov & Yashunin (2016), skip lists (Pugh, 1989), Kleinberg's navigable networks (Section 3.11).

### 8.3 Cheat Sheet: Advanced Retrievers for RAG

- Core advanced-retrieval concepts: semantic understanding, keyword matching, hierarchical context, multi-query processing, fusion (Section 4.1).
- Maximum Marginal Relevance (MMR) — relevance/diversity trade-off (Section 4.2).
- Full LlamaIndex index-type and retriever-type reference, including TF-IDF/BM25 math and all three Query Fusion strategies with formulas (Section 5).
- Full LangChain retriever reference: Vector Store-Backed, Multi-Query, Self-Querying, Parent Document (Section 6).
- The LlamaIndex-vs-LangChain decision framework table (Section 7).

---

## Appendix: Source Material Map

This reference was compiled from:
- **Video lessons**: Introduction to FAISS and how it compares to Chroma DB (including Milvus); Advanced Retrievers in LlamaIndex (index types, retriever types, fusion strategies).
- **Readings/cheat sheets**: *Cheat Sheet: Build a Comprehensive RAG Application* (FAISS vs. Chroma DB, FAISS index types, HNSW deep dive, Milvus); *Hierarchical Navigable Small World (HNSW)* (full conceptual + construction + search walkthrough); *Cheat Sheet: Advanced Retrievers for RAG* (LlamaIndex and LangChain retriever references, decision framework).
- **Code**: `Semantic_Similarity_with_FAISS.ipynb`.
- **Supporting files**: `lab-instructions (7).md`.
