# Vector Databases, Chroma DB & Similarity Search — Course Reference

*A consolidated study reference covering vector database fundamentals, vector database types and applications, how vector databases power RAG, Chroma DB architecture and operations, similarity search math, HNSW indexing, and Chroma DB filtering.*

> This is a companion reference to the earlier **"GenAI, LangChain & Flask"** and **"RAG, LlamaIndex & Gradio"** files. Concepts like RAG's 8-step pipeline and embeddings are introduced there and are extended here specifically in the context of vector databases and Chroma DB.

---

## Table of Contents

1. [Vector Database Fundamentals](#1-vector-database-fundamentals)
2. [Vector Databases vs. Traditional Databases](#2-vector-databases-vs-traditional-databases)
3. [Types of Vector Databases](#3-types-of-vector-databases)
4. [Vector Database Applications](#4-vector-database-applications)
5. [How Vector Databases Power RAG](#5-how-vector-databases-power-rag)
6. [Chroma DB — Key Concepts & Architecture](#6-chroma-db--key-concepts--architecture)
7. [Chroma DB — Essential Database Operations](#7-chroma-db--essential-database-operations)
8. [Similarity Search — The Underlying Math](#8-similarity-search--the-underlying-math)
9. [Vector Indexes & HNSW in Chroma DB](#9-vector-indexes--hnsw-in-chroma-db)
10. [Performing Similarity Searches in Chroma DB](#10-performing-similarity-searches-in-chroma-db)
11. [Chroma DB Filtering](#11-chroma-db-filtering)
12. [Cheat Sheets Recap](#12-cheat-sheets-recap)

---

## 1. Vector Database Fundamentals

### 1.1 Why Vector Databases?

Traditional relational databases have long been the standard for data management, but increasingly complex data types (images, audio, genomic data, social relationships) demand a more advanced solution. Companies use **vector databases** as libraries to find information, mine data, and teach computers to learn — simplifying data storage, organization, and retrieval by organizing data points in a **multi-dimensional space based on their proximity**.

Vector databases excel at handling **complex, non-traditional data** — relationship data (e.g., social likes), geospatial data, genomic data — that is difficult for traditional systems to store and manage without heavy pre-processing/transformation.

### 1.2 What Is a Vector?

A **vector** is a mathematical object defined by **size (magnitude) and direction**. Traditional relational databases store information as tables; a vector database instead stores data as **high-dimensional vectors** — an **array of numerical values** relating to different features/attributes of the data, where each numerical entry is a **dimension**.

**Worked example**: representing books as vectors `[genre, pages, publication_year, rating]`:
- Fiction book: `[1, 350, 2003, 4.5]`
- Non-fiction book: `[2, 250, 2015, 4.8]`
- Science-fiction book: `[3, 400, 1990, 4.2]`

Here, `1`=fiction, `2`=non-fiction, `3`=sci-fi (genre), followed by page count, publication year, and average rating. To find sci-fi books around 200 pages rated 4.7–5.0, you compare the **vector points** of candidate books directly rather than searching the whole platform — this is the essence of a **similarity search**.

### 1.3 What Vector Databases Enable

- **Similarity search** — quickly and accurately locate related items based on proximity in high-dimensional space (used for finding similar images/sounds, generating recommendations, genetic analysis).
- **Analytical tasks** — grouping items, classifying items, suggesting relationships among items.
- **Scalable big-data processing** — using distributed computing, indexing, and parallel processing to manage large datasets and process queries quickly, across domains like biology, healthcare, e-commerce, social media, and traffic planning.
- **ML/AI integration** — vector databases are a natural fit for storing and exploring machine learning data, integrating easily into ML pipelines and speeding up AI-powered app development.

### 1.4 Vector Database Fundamentals — Key Definitions

- **Vector Database**: a specialized database designed to store and query vectorized data rapidly; data is represented as vectors in multi-dimensional space, where each dimension corresponds to a specific attribute.
- **Vector Libraries vs. Vector Databases**:
  - **Vector Libraries** — in-memory, offer similarity search capabilities only.
  - **Vector Databases** — full **CRUD** (Create, Read, Update, Delete) operations, plus features suited to **enterprise-level production deployments**.

### 1.5 Creating Embeddings from Different Data Types

Images, text, and audio can each be passed through an appropriate **transformer** (Image Transformer, NLP Transformer, Audio Transformer) to produce vector embeddings (e.g., `{1.3, 0.4, ..., 0.4}` for an image), which are then stored in the vector database for downstream retrieval.

---

## 2. Vector Databases vs. Traditional Databases

### 2.1 How Relational Databases Organize Data

A relational database organizes data into **tables** with rows and columns. Each row = a record; each column = a property/attribute. Tables connect to each other via **keys** (primary and foreign keys) to represent relationships. Relational databases use **SQL** (`SELECT`, `INSERT`, `UPDATE`, `DELETE`) for querying/manipulation and excel at managing **structured data** with well-defined relationships (business/transactional systems).

### 2.2 Side-by-Side Comparison

| Function | Traditional Databases | Vector Databases |
|---|---|---|
| **Data Representation** | Structured tables, rows, columns — ideal for relational data | Multi-dimensional vectors — efficiently encode complex/unstructured data (images, text, sensor data) |
| **Data Search & Retrieval** | SQL queries for structured data | Similarity searches for vectorized data (image retrieval, recommendations, anomaly detection) |
| **Indexing** | B-trees for efficient retrieval | Specialized indices — e.g., **graph-based HNSW** — for approximate nearest neighbor search |
| **Scalability** | Resource augmentation or data sharding | Distributed architectures for horizontal scaling |
| **Applications** | Business/transactional systems | Context-aware AI apps, similarity search, NLP, multimedia analysis |

### 2.3 Key Takeaways

- Vector libraries **read and update** data; vector databases additionally support full **CRUD**.
- Vector databases store data as numerical vectors representing each data item.
- Relational databases organize data into rows/columns within tables.
- Traditional databases retrieve via **SQL queries**; vector databases retrieve via **similarity search**.

---

## 3. Types of Vector Databases

### 3.1 By Storage/Architecture Category

| Category | Description | Example Vendors |
|---|---|---|
| **In-memory** | Store vectors directly in memory for very fast read/write; ideal for real-time analytics and recommendation systems | **RedisAI**, Torchserve |
| **Disk-based** | Store vectors on disk; suited to datasets too large for memory; use sophisticated indexing/compression | **Annoy** (Approximate Nearest Neighbors Oh Yeah), **Milvus**, ScaNN |
| **Distributed** | Spread vector data across multiple nodes/servers for horizontal scalability and fault tolerance | **FAISS** (Facebook AI Similarity Search), Elasticsearch with Vector Plugin, Dask-ML |
| **Graph-based** | Model data as graphs (nodes/edges) representing vector attributes/embeddings; excel at complex relationships and graph analytics | **Neo4j**, Amazon Neptune, TigerGraph |
| **Time-series** | Manage data collected over time intervals, represented as vectors; analyze temporal patterns/anomalies | **InfluxDB**, TimescaleDB, Prometheus |

**Notable examples**:
- **RedisAI** — in-memory; supports similarity search, classification, clustering.
- **Annoy** — disk-based; builds indexes for fast retrieval; good for recommendation/information-retrieval systems.
- **FAISS** — distributed and optimized for similarity search in high-dimensional spaces, partitioning data across nodes.
- **Neo4j** — graph-based; supports storing vectors as node properties; used for social network analysis, recommendation systems, knowledge graphs.
- **InfluxDB** — allows vector storage alongside time-stamped data; used for IoT, monitoring, anomaly detection.

### 3.2 Dedicated Vector Databases vs. Databases That Support Vector Search

**Dedicated vector databases** are purpose-built systems with their own optimizations for storing, indexing, querying, and analyzing vector data at scale.

- Use specialized data structures: **inverted indexes, product quantization, locality-sensitive hashing (LSH)**.
- Support core vector operations: nearest-neighbor search, similarity search, distance calculations.
- Built for **scalability** across clusters/distributed systems.
- Prioritize **speed** via optimized algorithms/data structures, even for high-dimensional data.
- Allow users to tune indexing/search parameters for specific use cases.
- Popular examples: **FAISS, Annoy, Milvus**.

**Databases that support vector search** are regular database systems or data-processing frameworks that add vector capabilities via extensions/plugins, without being purpose-built for vector operations.

- Store vector data as blobs, arrays, or user-defined types (UDTs).
- May offer custom/standard index structures for retrieving vectors by similarity/distance.
- Often rely on add-ons/plugins/external libraries for vector-related tasks.
- Generally **less optimized/fast** than dedicated vector databases.
- Popular vendors: **SingleStore** (works with IBM watsonx.ai), **Elasticsearch** (vector add-on), **PostgreSQL** (PostGIS add-on), **MySQL** (vector indexes), **RedisAI**, **Apache MongoDB**, **Apache Cassandra** (flexible-schema vector search).

---

## 4. Vector Database Applications

### 4.1 Image & Video Analysis

Three core capabilities:
- **Feature extraction & representation** — store high-dimensional feature vectors capturing color histograms, texture descriptions, or deep-learning embeddings.
- **Similarity search** — locate images, summarize videos, suggest content based on visual similarity.
- **Real-time processing** — horizontal scalability enables video surveillance, object recognition, and live event analysis.

**Example**: a photo-sharing app stores embeddings of user photos; when a new photo is added, the app compares its embedding to others in the database and, if similar, suggests it for tagging/organizing into albums.

### 4.2 Recommendation Systems

- **Embedded storage & nearest-neighbor search** — numerical embeddings of items/entities power personalized suggestions based on likes/traits.
- **Performance & scalability** — vector databases scale query processing/indexing to serve fast recommendations to many concurrent users.
- **Cross-domain suggestions** — embeddings enable recommendations that span domains, improving completeness.

**Example**: a streaming service stores movie embeddings; after you watch a movie, the system uses embeddings of related movies to recommend what to watch next.

### 4.3 Geospatial Analysis & Location-Based Services

- **Efficient storage/indexing** — geospatial data (addresses, polygons, GPS locations) indexed with structures like **R-tree** or **quadtree**, enabling closeness searches, range queries, and spatial joins.
- **Location-based suggestions** — combine geospatial data with user preferences to suggest nearby events, services, places.
- **Real-time geospatial analytics** — process streaming location data, cluster items spatially, detect spatial patterns — powering vehicle tracking, fleet management, dynamic routing, hotspot detection.

**Example**: a navigation app stores GPS locations of restaurants in a vector database and returns a list of nearby options within a chosen distance.

### 4.4 Social & Marketing Insights

- **Distributed storage & parallel processing** — horizontal scalability across nodes/clusters lets platforms process big data and handle simultaneous queries (SEO calculations, user profile management).
- **Optimized caching & query execution** — reduce latency, speeding delivery of trending analytics to influencers/advertisers.
- **Autoscaling & dynamic resource allocation** — adapt hardware/cloud usage to changing workload for the best performance/cost trade-off.

**Example**: a social platform tracks user hobbies/interests (cycling, running, swimming) and product-click behavior; as the user base grows, it scales hardware to keep response times fast.

---

## 5. How Vector Databases Power RAG

### 5.1 Recap: What RAG Solves

RAG enhances language models by **retrieving relevant information from external sources** and using it to generate more accurate, grounded responses — reducing hallucinations. This addresses two core LLM limitations:
- **Limited context windows** — not all information can fit in a single prompt.
- **Frozen/hallucination-prone knowledge** — models' knowledge is fixed at training time and can produce fabricated facts.

### 5.2 The Full RAG Pipeline (as implemented with a vector database)

1. Provide relevant **source documents**, potentially split into smaller **chunks**.
2. **Embed** the source documents/chunks.
3. **Store** the sources and their embeddings in a vector database (e.g., Chroma DB).
4. **Receive** the user's prompt.
5. **Embed** the user's prompt.
6. A **retriever** selects the chunks from the vector store that best match the prompt.
7. **Combine** the retrieved text with the original prompt → an augmented prompt.
8. **Pass** the augmented prompt to the LLM to produce a context-aware response.

### 5.3 Vector Database Responsibilities in RAG

Vector databases can handle:
- **Embedding** both source documents and user prompts,
- **Storing** those embeddings,
- **Retrieving** the most relevant matches,
- **Supplying** the retrieved content for prompt augmentation.

Note: steps 2 and 5 (embedding) can also be done **externally**; in that case the vector database is used primarily just for storing/retrieving vectors.

### 5.4 Why Use a Vector Database for These Steps?

1. **Prevents critical mistakes** — e.g., accidentally using different embedding models for documents vs. queries, or mis-linking embeddings to their source documents.
2. **Faster, cleaner development** — offloading steps to the vector database means fewer moving parts and less custom logic, keeping the codebase simpler and easier to debug.
3. **Performance** — vector databases are purpose-built for high-speed, scalable semantic search using advanced indexing algorithms; custom-built alternatives typically can't match this without significant optimization effort.

### 5.5 Common RAG Pipeline Pitfalls

| Pitfall | Why it happens | Solution |
|---|---|---|
| **Different embedding models** for documents vs. queries | Breaks retrieval entirely | Use the same embedding model throughout — vector databases usually enforce this automatically |
| **Poor chunking strategy** | Chunks too large or too small hurt retrieval quality | Choose a chunk size long enough to preserve meaning without pulling in irrelevant content |
| **Forgetting to re-embed** after changing data, distance metric, or embedding model | Stale embeddings no longer reflect the model/metric in use | For some databases (e.g., Chroma DB) this can't be done on an existing collection — you may need to **clone the collection** |
| **Assuming the retrieved result is always the best answer** | Retrieval isn't infallible | Always test your results — a little tuning can make a big difference |

### 5.6 What Vector Databases Don't Handle

Some RAG tasks typically happen **outside** the vector database:
- **Chunking** — usually done before data enters the database.
- **Extra retrieval logic** (filtering, re-ranking) — may need additional tools.
- **Prompt augmentation** — typically handled outside the database.
- **LLM integration** — not built into most vector databases.

**RAG frameworks** like **LangChain** and **LlamaIndex** wrap around the vector database to manage the full pipeline (document prep → final response), providing additional structure and simplifying development/deployment.

---

## 6. Chroma DB — Key Concepts & Architecture

### 6.1 Core Capabilities

Chroma DB is a vector database purpose-built for retrieval tasks. It offers:
- **Storage of embeddings and metadata** — efficiently stores/manages vector representations plus associated metadata.
- **Vector search** — compares embeddings to find text based on semantic similarity, using distance metrics (e.g., cosine distance).
- **Full-text search** — finds relevant documents based on lexical/spelling similarity.
- **Data storage** — stores entire documents, not just their embeddings.
- **Metadata filtering** — narrows search results based on metadata to improve retrieval accuracy.
- **Multi-modal retrieval** — retrieve/manage multi-modal data (images, audio, text) in a unified way.

### 6.2 Deployment Options

- **Client-server architecture** (typical): a Chroma client connects over HTTP to a Chroma server running in a separate process; the server is launched via the Chroma CLI (core Chroma package) or a Docker image.
- **Standalone mode** (Python only): server and client functionality run in a single process — useful for quickly testing features or when server and client always run on the same machine.

### 6.3 Architecture Phases

1. **Obtain embeddings** *(optional)* — convert text/images/data into vector representations using an embedding model. Optional because you can offload embedding to Chroma DB itself.
2. **Create collections** — similar to tables in a relational database; Chroma DB uses collections to organize all its data.
3. **Store data within collections** — pass in precomputed embeddings, or let Chroma DB calculate and store them automatically in the background.
4. **Perform collection operations** — delete, update, or rename collections.
5. **Query and group data** — use text or vector queries to find information based on semantic meaning or textual similarity; filter on metadata and document contents.

### 6.4 Clients & Integrations

- **Officially supported clients** (maintained by ChromaCore): **Python** and **JavaScript**.
- **Community-supported clients**: Ruby, Java, Go, C#, Rust, PHP (see the Chroma Ecosystem Clients page in the Chroma Cookbook for details).
- **Framework integrations**: LangChain, LlamaIndex, Ollama.
- **Native embedding model integrations**: Hugging Face, Google, OpenAI.

### 6.5 A Typical Chroma DB Workflow

1. **Create a collection** (give it a logical name).
2. **Add** chunks of text + associated metadata — Chroma DB automatically stores the text and handles embedding (or you supply precomputed embeddings).
3. **Query** the collection — Chroma DB returns the most similar results, automatically embedding your query text too (no need to embed the query yourself).

By default, Chroma DB uses **Euclidean (L2) distance** to find the most similar chunks; it also supports **cosine distance** and **dot product**.

### 6.6 Performance Characteristics

- Optimized for **approximate nearest neighbor (ANN) search** using the **HNSW** algorithm (see Section 9).
- Core written in **Rust**, giving **3–5x speed improvements** in querying/writing versus a pure-Python core.

### 6.7 Common Use Cases

- Personalized **recommender systems** based on user preferences.
- Efficient **document search engines** (vector or full-text search).
- **Image retrieval** based on text queries (multi-modal retrieval).
- **Chatbots** with semantic search/retrieval for context augmentation.

---

## 7. Chroma DB — Essential Database Operations

### 7.1 Creating Collections

```python
import chromadb
from chromadb.utils import embedding_functions

# Define the embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Define Chroma DB client
client = chromadb.Client()

# Create a collection
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "A collection for storing user data"},
    configuration={
        "embedding_function": sentence_transformer_ef
    }
)
print(f"Collection created: {collection.name}")
```
- Metadata (e.g., `description`) helps track a collection's purpose/contents and can hold **any key-value pair**.

### 7.2 Connecting to an Existing Collection

```python
collection = client.get_collection(name="my_collection")
```

### 7.3 Modifying Collections

```python
collection.modify(
    name="new_collection_name",
    metadata={"key": "value"}
)
```
**Important**: certain changes — like the **embedding model** or **distance metric** — **cannot** be made to an existing collection. To apply such changes, you must **clone the collection**, which can be computationally expensive for large collections.

### 7.4 Adding Documents

```python
collection.add(
    documents=[
        "This is a document about LangChain",
        "This is a document about LlamaIndex"
    ],
    metadatas=[
        {"source": "langchain.com", "version": "0.2"},
        {"source": "llamaindex.ai", "version": "0.12"}
    ],
    ids=["id1", "id2"]
)
```
- `documents` — list of texts to insert.
- `metadatas` — optional list of dicts, one per document (no restrictions on content).
- `ids` — **required**; a unique ID for each document.

### 7.5 Retrieving Documents

```python
# Get all documents (returns a Python dict)
results = collection.get()

# Get specific documents by ID
results = collection.get(ids=["id1"])

# Include embeddings in results (excluded by default to keep output clean)
results = collection.get(include=['embeddings'])
```
Embeddings *are* stored in the collection even though `get()` doesn't return them by default.

### 7.6 Updating Documents

```python
collection.update(
    ids=["id1"],
    metadatas=[{"source": "langchain.com", "version": "0.3"}],
    documents=["This an updated document about LangChain"]
)
```
Chroma DB automatically **re-embeds** the document in the background as soon as the update is submitted.

### 7.7 Deleting Documents

```python
# By ID
collection.delete(ids=["id1"])

# By metadata filter
collection.delete(where={"source": "doc_to_delete.pdf"})

# Combine IDs and filters
collection.delete(ids=["id1"], where={"version": "1.0"})
```

### 7.8 Distance Functions & HNSW `space` Parameter

Chroma DB uses **HNSW (Hierarchical Navigable Small World)** for approximate nearest-neighbor search. The **`space`** parameter (set at collection-creation time) defines the distance function:

| Value | Meaning | Default? |
|---|---|---|
| `l2` | Squared L2 norm (Euclidean distance) | ✅ Default |
| `cosine` | Cosine distance | |
| `ip` | Inner product / dot product distance | |

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "A collection for storing user data"},
    configuration={
        "embedding_function": sentence_transformer_ef,
        "hnsw": {"space": "cosine"}
    }
)
```

---

## 8. Similarity Search — The Underlying Math

### 8.1 What Is Similarity Search?

**Similarity search** is the process of finding items in a dataset that are most similar to a given query item. Widely used in:
- **Recommendation systems** (e.g., suggesting similar movies)
- **Image and video retrieval**
- **NLP** (e.g., finding similar documents/sentences)
- **Biometrics** (e.g., face recognition)

At its core is a **distance or similarity metric** quantifying how alike two data points are; the right choice depends on the data and the application.

### 8.2 Background: Vectors, Magnitude, and the Cosine of an Angle

- A **vector** has length and direction; a 2-component vector `a = [4, 8]` can be drawn on a 2D Cartesian plane; a 3-component vector needs 3D, and so on.
- **Magnitude (L2 norm)** of a vector: `||a|| = √(Σ aₖ²)`. For `a = [4, 8]`: `||a|| = √(4² + 8²) ≈ 8.94`.
- **Cosine of an angle**: `cos(α) = adjacent / hypotenuse`. In vector terms, for the angle α between vectors `a` and `b`, this relationship underlies both the dot product and cosine similarity formulas below.
- When vectors are used as **embeddings**, the **direction** typically encodes semantic meaning/topic, while the **magnitude** can reflect intensity, confidence, or salience (e.g., how popular a product is, or how authoritative a source is).

### 8.3 L2 Distance (Euclidean Distance)

**Definition**: `L2(a, b) = √(Σᵢ (aᵢ − bᵢ)²)`

**Example**: for `a = [4, 8]`, `b = [11.5, 5]`:
`L2(a, b) = √((4−11.5)² + (8−5)²) ≈ 8.08`

**Properties**
- Measures straight-line distance between two points (Pythagorean theorem).
- Sensitive to **both magnitude and direction**.
- Common in spatial/geometric applications: image analysis, computer vision, geographic mapping.

**Use case**: finding the closest point to a location in 2D/3D space — common in computer vision.

### 8.4 Dot Product (Inner Product) Similarity

**Definition**: `a · b = Σᵢ aᵢbᵢ`

**Example**: for `a = [4, 8]`, `b = [11.5, 5]`: `a · b = 4×11.5 + 8×5 = 86`

**Alternative calculation (via magnitudes and angle)**: `a · b = ||a|| ||b|| cos(α)`
- For `a = [4, 8]` (`||a|| ≈ 8.94`), `b = [11.5, 5]` (`||b|| ≈ 12.54`), and `α ≈ 39.94°` (`cos ≈ 0.767`): `a · b ≈ 8.94 × 12.54 × 0.767 ≈ 85.99` ✓ (matches, allowing for rounding).

**Properties**
- Can be **positive, negative, or zero**, depending on the angle between vectors.
- **Larger** values indicate **more similar** direction (higher similarity).
- To use as a **distance** metric, take the **negative** of the dot product (larger-negative = more distant).
- Sensitive to both magnitude and direction.
- Common in neural network activations and matrix factorization for recommender systems.

**Use case**: when a vector's length carries meaning (relevance, confidence, popularity) — e.g., recommending items that are both topically similar **and** popular.

### 8.5 Cosine Similarity & Distance

**Definition**: `cosine_similarity(a, b) = (a · b) / (||a|| ||b||)`

**Distance conversion**: `cosine_distance(a, b) = 1 − cosine_similarity(a, b)`

**Efficient calculation with normalized vectors**: normalize a vector via `norm(a) = a / ||a||` (a normalized vector's squared components sum to 1). Once normalized, cosine similarity is just the dot product: `cosine_similarity(a, b) = norm(a) · norm(b)`. Many embedding models normalize vectors by default for exactly this reason — cosine comparisons become as cheap as a dot product.

**Properties**
- Focuses on **orientation**, not magnitude — measures the angle between vectors.
- Well-suited to **high-dimensional, sparse data** (text embeddings, term-frequency vectors).
- **Invariant to vector length** — scaling a vector doesn't change the similarity score.

**Use case**: measuring document similarity in NLP, regardless of document length.

### 8.6 Choosing the Right Metric

| Metric | Sensitive to Magnitude | Normalized | Best For |
|---|---|---|---|
| **L2 Distance** | ✅ Yes | ❌ No | Spatial data, clustering |
| **Cosine Distance** | ❌ No | ✅ Yes | Text, embeddings, NLP |
| **Dot Product** | ✅ Yes | ❌ No | Neural networks, recommender systems |

**Practical considerations**
- **Normalize vectors** if you'll only ever need cosine similarity — the default for many NLP/text-embedding tasks.
- **High-dimensional data**: L2 distance can suffer from the "curse of dimensionality" — consider a different metric or dimensionality reduction. Cosine distance often performs better in high dimensions and is the default in many NLP/text tasks.
- Dot product computes efficiently via matrix operations and doubles as cosine similarity when vectors are normalized.

---

## 9. Vector Indexes & HNSW in Chroma DB

### 9.1 What Is a Vector Index?

Computing exact similarity (e.g., cosine similarity via normalized dot products) is mathematically simple — but comparing a query against **every** vector in the database (brute force) becomes slow at scale.

A **vector index** is a specialized data structure that organizes high-dimensional embeddings to reflect the geometry of the vector space (clustering similar vectors together, or linking them via proximity-based graphs). This lets the search algorithm **prune** large portions of the dataset early, enabling scaling to millions or billions of vectors while keeping low latency.

### 9.2 What Is HNSW?

**HNSW (Hierarchical Navigable Small World)** is a fast, scalable, graph-based vector index for **approximate nearest neighbor (ANN)** search in high-dimensional spaces. It is the **sole indexing method supported by Chroma DB**, and is widely adopted elsewhere for its performance/reliability.

**How it works**
- Builds a **multi-layered graph**:
  - **Upper layers** — a sparse overview of the data for fast navigation.
  - **Bottom layer** — holds all vectors for detailed search.
- Each vector connects to a few nearby neighbors, forming a "small world" network — most vectors can be reached in just a few hops.

**Search process**: the algorithm starts at the top layer and descends, getting closer to the query vector at each level, refining the search — skipping most of the dataset while still finding highly similar vectors.

**Why HNSW?**
- **Fast** — avoids scanning the entire dataset.
- **Accurate** — near-exact results.
- **Scalable** — millions to billions of vectors.
- **Versatile** — works with multiple similarity metrics.

### 9.3 Configuring HNSW in Chroma DB

HNSW is configured **at collection-creation time** via the `hnsw` key:

```python
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()
collection = client.create_collection(
    name="my_collection_name",
    metadata={"topic": "query testing"},
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef_search": 100,
            "ef_construction": 100,
            "max_neighbors": 16
        },
        "embedding_function": ef
    }
)
```

**Key parameters**

| Parameter | Meaning | Default | Trade-off |
|---|---|---|---|
| **`space`** | Distance metric: `l2` (default), `ip`, or `cosine` | `l2` | — |
| **`ef_search`** | Candidate-list size used when searching for nearest neighbors at query time | 100 | Higher → better accuracy/recall, but slower & more compute |
| **`ef_construction`** | Candidate-list size used when selecting neighbors while inserting a node during index build | 100 | Higher → better index quality/accuracy, but slower build & more memory |
| **`max_neighbors`** | Max connections per node during construction | 16 | Higher → denser graph, better search performance, but more memory & longer build time |

**Two categories of tuning**
- **`ef_search`** directly controls query-time search breadth — the most direct lever for **recall vs. query speed**.
- **`ef_construction`** and **`max_neighbors`** affect the **quality of the built index** itself — a higher-quality, denser index (from higher values) gives a better foundation for search, at the cost of significantly longer build times and higher memory use during construction and storage.

---

## 10. Performing Similarity Searches in Chroma DB

### 10.1 Adding Data for a Worked Example

```python
collection.add(
    documents=[
        "Giant pandas are a bear species that lives in mountainous areas.",
        "A pandas DataFrame stores two-dimensional, tabular data",
        "I think everyone agrees that pandas are some of the cutest animals on the planet",
        "A direct comparison between pandas and polars indicates that polars is a more efficient library than pandas.",
    ],
    metadatas=[
        {"topic": "animals"},
        {"topic": "data analysis"},
        {"topic": "animals"},
        {"topic": "data analysis"},
    ],
    ids=["id1", "id2", "id3", "id4"]
)
```
All four documents mention "pandas," but two refer to the **animal** and two to the **Python library** — a good test set for whether semantic search can distinguish meanings by context.

### 10.2 Basic Query

```python
collection.query(
    query_texts=["cats"],
    n_results=10,
)
```
- `query_texts` — the query (or queries), passed as a list.
- `n_results` — how many results to retrieve (if it exceeds the collection size, all documents are returned, ranked most→least similar).

**Result for `"cats"`** (excerpted): the top two matches were the two **animal**-topic documents — the "cutest animals" line ranked highest (lowest cosine distance), likely because "cats" and "cute" are semantically close, followed by the general panda-habitat sentence. The two "pandas-the-library" documents ranked lowest.

### 10.3 When Semantic Search Goes Wrong — and Fixing It with Filters

Querying `"polar bear"` with `n_results=1` (no filter) can **fail**: the word "polar" gets matched to "**polars**" (the Python library) purely on embedding proximity, returning the wrong document entirely.

**Fix 1 — metadata filter**:
```python
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={'topic': 'animals'}
)
```
This correctly narrows results to the bear-related document.

**Fix 2 — document/full-text filter** (exclude documents containing a word):
```python
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where_document={'$not_contains': 'library'}
)
```

**Fix 3 — combine both**:
```python
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={'topic': 'animals'},
    where_document={'$not_contains': 'library'}
)
```

**Other mitigation options**: refine the query with more context, or try a different embedding model that better captures the intended meaning.

### 10.4 Common Workflow Pattern (End-to-End)

```python
# 1. Setup
import chromadb
from chromadb.utils import embedding_functions

# 2. Create embedding function and client
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()

# 3. Create collection with configuration
collection = client.create_collection(
    name="collection_name",
    configuration={"hnsw": {"space": "cosine"}, "embedding_function": ef}
)

# 4. Add documents
collection.add(documents=texts, metadatas=metadata, ids=ids)

# 5. Perform similarity search
results = collection.query(query_texts=["query"], n_results=5)

# 6. Process results
for i, (doc_id, score, text) in enumerate(zip(results['ids'][0], results['distances'][0], results['documents'][0])):
    print(f"Rank {i+1}: {doc_id}, Score: {score:.4f}, Text: {text}")
```

### 10.5 Full Grocery Example (Multi-Query Similarity Search)

```python
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()
collection_name = "my_grocery_collection"

collection = client.create_collection(
    name=collection_name,
    metadata={"description": "Grocery Data Collection"},
    configuration={
        'hnsw': {'space': 'cosine'},
        'embedding_function': ef,
    }
)

def main():
    try:
        texts = [
            'fresh red apples', 'organic bananas', 'ripe mangoes', 'whole wheat bread',
            'farm-fresh eggs', 'natural yogurt', 'frozen vegetables', 'grass-fed beef',
            'free-range chicken', 'fresh salmon fillet', 'aromatic coffee beans',
            'pure honey', 'golden apple', 'red fruit'
        ]
        ids = [f"food_{index+1}" for index, _ in enumerate(texts)]
        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids,
        )
        all_items = collection.get()
        perform_similarity_search(collection, all_items)
    except Exception as error:
        print(f"Error: {error}")

def perform_similarity_search(collection, all_items):
    query_term = ['red', 'fresh']
    results = collection.query(query_texts=query_term, n_results=3)
    for q in range(len(query_term)):
        print(f'Top 3 similar documents to "{query_term[q]}":')
        for i in range(min(3, len(results['ids'][q]))):
            doc_id = results['ids'][q][i]
            score = results['distances'][q][i]
            text = results['documents'][q][i]
            print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')

if __name__ == "__main__":
    main()
```
This demonstrates **multi-query** similarity search — `query_texts` can hold several queries at once, with `results['ids']`, `['distances']`, and `['documents']` each being a **list of lists** (one inner list per query, in the same order as `query_texts`).

---

## 11. Chroma DB Filtering

### 11.1 Why Filtering Is Different in Chroma DB

Chroma DB filtering fundamentally differs from SQL-based filtering because it's built around **vector similarity** and **flexible metadata querying**, rather than structured schemas and exact-match declarative logic — making it well suited to unstructured data and semantic search.

Chroma DB supports **two primary filter types**:

| Filter Type | Description | SQL Analogy |
|---|---|---|
| **Metadata Filtering** | Filters on document metadata (e.g., `"topic": "history"`, `"date": "2023-01-15"`) | Like `WHERE`, but more flexible and combinable with vector search |
| **Document Filtering** | Filters on document content via keyword presence (`$contains`, `$not_contains`) | Like `CONTAINS`/`LIKE`, but more powerful when integrated with vector search |

Document filtering is also called **full text search** in Chroma DB.

### 11.2 Metadata Filtering

Applied via the `where` parameter inside `.query()`, `.get()`, or `.delete()`.

**Basic equality**:
```python
collection.get(where={"key": "value"})
# Equivalent to:
collection.get(where={"key": {"$eq": "value"}})
```
Not specifying an operator is equivalent to using `$eq`.

**Comparison operators**:

| Operator | Meaning | Applies to |
|---|---|---|
| `$eq` | equal to | string, int, float |
| `$ne` | not equal to | string, int, float |
| `$gt` | greater than | int, float |
| `$gte` | greater than or equal to | int, float |
| `$lt` | less than | int, float |
| `$lte` | less than or equal to | int, float |
| `$in` | value is in a list | any |
| `$nin` | value is **not** in a list | any |

**Logical combination (`$and` / `$or`)**:
```python
collection.get(
    where={
        "$and": [
            {"source": {"$eq": "langchain.com"}},
            {"version": {"$lt": 0.3}}
        ]
    }
)
```
`$or` works the same way. `$and`/`$or` also work inside `.query()` and `.delete()`.

**List-based filtering with `$in`**:
```python
collection.get(
    where={
        "$and": [
            {"source": {"$in": ["langchain.com", "llamaindex.ai"]}},
            {"version": {"$lt": 0.3}}
        ]
    }
)
```

### 11.3 Document (Full-Text) Filtering

Applied via the `where_document` parameter inside `.query()`, `.get()`, or `.delete()`.

```python
# Contains text
where_document={"$contains": "pandas"}

# Does not contain text
where_document={"$not_contains": "library"}

# Combined with logical operators
where_document={
    "$or": [
        {"$contains": "LangChain"},
        {"$contains": "Python"}
    ]
}
```
**Important**: document filtering is **case-sensitive** — searching for `"Pandas"` will **not** match `"pandas"`.

### 11.4 Full Worked Example

```python
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()

collection = client.create_collection(
    name="filter_demo",
    metadata={"description": "Used to demo filtering in ChromaDB"},
    configuration={"embedding_function": ef}
)

collection.add(
    documents=[
        "This is a document about LangChain",
        "This is a reading about LlamaIndex",
        "This is a book about Python",
        "This is a document about pandas",
        "This is another document about LangChain"
    ],
    metadatas=[
        {"source": "langchain.com", "version": 0.1},
        {"source": "llamaindex.ai", "version": 0.2},
        {"source": "python.org", "version": 0.3},
        {"source": "pandas.pydata.org", "version": 0.4},
        {"source": "langchain.com", "version": 0.5},
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)

# All LangChain documents
collection.get(where={"source": {"$eq": "langchain.com"}})
# -> id1, id5

# LangChain documents with version < 0.3
collection.get(where={"$and": [{"source": {"$eq": "langchain.com"}}, {"version": {"$lt": 0.3}}]})
# -> id1

# LangChain or LlamaIndex documents with version < 0.3
collection.get(where={"$and": [{"source": {"$in": ["langchain.com", "llamaindex.ai"]}}, {"version": {"$lt": 0.3}}]})
# -> id1, id2

# Full-text search for "pandas"
collection.get(where_document={"$contains": "pandas"})
# -> id4 (only the exact-case match)

# Combine metadata + document filters: "LangChain" or "Python" mentions with version > 0.1
collection.get(
    where={"version": {"$gt": 0.1}},
    where_document={"$or": [{"$contains": "LangChain"}, {"$contains": "Python"}]}
)
# -> id3, id5
```

### 11.5 Filtering Best Practices

- Document filtering is **case-sensitive**.
- Combine metadata and document filters (`where` + `where_document`) for precise, context-aware results.
- Use `$and` / `$or` for complex filtering logic.
- Use `$in` / `$nin` for list-based filtering.

---

## 12. Cheat Sheets Recap

### 12.1 Cheat Sheet: Vector Databases for Recommendation Systems and RAG

- RAG's core problems solved: limited context windows, frozen training-time knowledge, hallucination.
- The 8-step RAG pipeline (Section 5.2) and vector-database responsibilities within it (embedding, storing, retrieving, supplying content for augmentation).
- Three reasons to use a vector database for RAG steps: prevents critical mistakes, faster/cleaner development, superior performance.
- Common pitfalls: mismatched embedding models, poor chunking, forgetting to re-embed after changes, blindly trusting retrieval results.
- What's outside the vector database: chunking, advanced retrieval logic, prompt augmentation, LLM integration — filled in by frameworks like LangChain/LlamaIndex.
- Full Chroma DB operations reference: create/connect/modify collections, add/get/update/delete documents, HNSW distance-function configuration (Section 7).

### 12.2 Cheat Sheet: Introduction to Vector Databases and Chroma DB

- Full math reference for **L2 distance, dot product, and cosine similarity/distance**, with properties and use cases (Section 8).
- Metric selection table: sensitivity to magnitude, normalization, best-fit use case.
- Practical considerations: normalization defaults, curse of dimensionality, matrix-efficient dot products.
- Vector databases vs. traditional databases comparison table (Section 2).
- Vector libraries vs. vector databases (in-memory + similarity only, vs. full CRUD + production deployment).
- Chroma DB best practices: case-sensitive document filtering, combining `where`/`where_document`, `$and`/`$or`/`$in`/`$nin` operators.
- What a **vector index** is, and why brute-force search doesn't scale.
- What **HNSW** is, how it works (multi-layered graph), and why to use it.
- HNSW configuration parameters: `space`, `ef_search`, `ef_construction`, `max_neighbors`, and their performance trade-offs.
- Full Chroma DB setup, collection creation, data operations, filtering syntax, and a complete common-workflow code pattern (Sections 7, 9, 10, 11).

### 12.3 Reading: Similarity Search and HNSW in Chroma DB

- Deep dive into vector indexes and HNSW (Section 9).
- Full worked example: adding ambiguous "pandas" documents (animal vs. library), querying `"cats"` and `"polar bear"`, diagnosing why "polar" matched "polars," and fixing it with metadata and document filters (Section 10).

### 12.4 Reading: Similarity Search (Math Foundations)

- Background math: cosine of an angle, what a vector is, vector magnitude/L2 norm, plotting multiple vectors.
- Full derivations of L2 distance, dot product (including the geometric/projection-based alternative calculation), and cosine similarity/distance, with normalized-vector shortcuts (Section 8).

### 12.5 Reading: Vector Databases Versus Traditional Databases

- Vector database data storage (numerical vectors, one dimension per attribute) vs. relational storage (tables, rows, columns, keys).
- Vector libraries (read/update only) vs. vector databases (full CRUD, enterprise production use).
- Full comparison table across data representation, search/retrieval, indexing, scalability, and applications (Section 2.2).

### 12.6 Reading: Chroma DB Filtering

- Dual-filtering approach: metadata filtering (`where`) vs. document/full-text filtering (`where_document`).
- Full metadata operator reference (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`) and logical combination (`$and`, `$or`).
- Complete worked example combining both filter types (Section 11.4).

---

## Appendix: Source Material Map

This reference was compiled from:
- **Video lessons**: How Vector Databases Power RAG; Essential Database Operations in Chroma DB; Chroma DB Key Concepts and Architecture; Exploring vector database applications (image/video, recommendations, geospatial, social/marketing); Types of Vector Databases; Essential Vector Database Concepts (vectors, book-recommendation example).
- **Readings/cheat sheets**: *Vector Databases for Recommendation Systems and RAG Cheat Sheet*; *Introduction to Vector Databases and Chroma DB Cheat Sheet*; *Similarity Search and HNSW in Chroma DB*; *Chroma DB Filtering*; *Similarity Search* (math foundations); *Vector Databases Versus Traditional Databases* (author: Richa Arora).
- **Code**: `similarity_search.py` (grocery-collection similarity search example), `Similarity_Search_by_Hand.ipynb`.
- **Supporting files**: `lab-instructions (4).md` / `(5).md` / `(6).md`.
