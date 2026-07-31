# 🕸️ Graphitti — Graph-Native Web Intelligence

Graphitti turns any set of web pages into a queryable **knowledge graph**. It crawls URLs, extracts entities and relationships with NLP, stores the facts in **Neo4j**, indexes the raw text in a **vector store (ChromaDB)** for semantic search, and answers natural-language questions using a **hybrid graph + vector retrieval** pipeline — all wrapped in an interactive Streamlit UI.

> Built as an academic project to demonstrate an end-to-end knowledge graph construction and retrieval-augmented question answering (RAG) pipeline.

---

## ✨ Features

- **Web crawling** — scrapes and cleans text from any list of URLs (e.g. Wikipedia pages)
- **Entity & relationship extraction** — uses spaCy dependency parsing to pull `(subject, relation, object)` triplets and named entities out of raw text
- **Knowledge graph construction** — writes entities and relationships into a **Neo4j** graph database
- **Interactive visualization** — renders the graph as an explorable, physics-based network with [pyvis](https://pyvis.readthedocs.io/) (color/shape-coded by entity type)
- **Semantic search** — chunks and embeds crawled text with `sentence-transformers/all-MiniLM-L6-v2`, stored in **ChromaDB**
- **Query classification** — an LLM (Llama 3 via Ollama) decides whether a question needs graph facts, vector context, or both
- **Hybrid RAG Q&A** — combines Cypher query generation (with automatic self-repair on failure) and vector similarity search to answer questions grounded strictly in retrieved context
- **Streamlit web app** — a single-page UI to crawl, build, visualize, and chat with the graph, plus a CLI (`main.py`) for terminal use

---

## 🧠 How it works

```
URLs
  │
  ▼
crawler.py            → scrapes & cleans page text
  │
  ▼
nlp.py                 → spaCy NER + dependency parsing → entities & triplets
  │
  ├──────────────────────────────┐
  ▼                              ▼
graph_pipeline.py         embeddings.py + chroma.py
(writes triplets              (chunks text, embeds it,
 to Neo4j)                     stores in ChromaDB)
  │                              │
  ▼                              │
visualization.py                 │
(renders graph.html              │
 with pyvis)                     │
                                  │
        Question ────────────────┘
          │
          ▼
  query_classifier.py   → GRAPH / VECTOR / HYBRID
          │
          ▼
  hybrid_retriever.py    → graph_retriever.py (NL → Cypher via LLM)
                           vector_retriever.py (ChromaDB similarity search)
          │
          ▼
  answer_generator.py    → LLM answers using only retrieved context
          │
          ▼
      Final answer
```

The whole thing is orchestrated as a small [LangGraph](https://langchain-ai.github.io/langgraph/) state machine (`langgraph_pipeline.py`) with two nodes: **retrieve** → **generate**.

---

## 🏗️ Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend — crawl, build graph, visualize, and chat in one page |
| `main.py` | CLI entry point — runs the full pipeline end-to-end, then a terminal Q&A loop |
| `crawler.py` | Fetches and cleans text from a list of URLs |
| `nlp.py` | Extracts named entities and subject–relation–object triplets with spaCy |
| `graph_pipeline.py` | Writes triplets into Neo4j as `(:Entity)-[:RELATION]->(:Entity)` |
| `visualization.py` | Reads the graph from Neo4j and renders it as `graph.html` with pyvis |
| `embeddings.py` | Wraps a HuggingFace sentence-transformer model for embedding text |
| `chroma.py` | Manages the ChromaDB vector store (add, search, delete) |
| `vector_retriever.py` | Semantic similarity search over ChromaDB |
| `graph_retriever.py` | Translates a question into Cypher via an LLM, runs it, repairs it on failure |
| `hybrid_retriever.py` | Merges graph and vector context, deduplicating overlaps |
| `query_classifier.py` | Classifies a question as `GRAPH`, `VECTOR`, or `HYBRID` |
| `answer_generator.py` | Generates the final answer strictly from retrieved context |
| `langgraph_pipeline.py` | Wires retrieval + generation into a LangGraph state graph |
| `test_connection.py` | Quick script to verify Neo4j connectivity |
| `config.py` | Neo4j connection settings and spaCy model config |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Prerequisites

- **Python 3.10+**
- A **Neo4j** database — either [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) (free tier works) or a local instance
- **[Ollama](https://ollama.com/)** installed and running locally, with the Llama 3 model pulled:
  ```bash
  ollama pull llama3
  ```
  (used for query classification, Cypher generation, and answer generation)
- The spaCy English model:
  ```bash
  python -m spacy download en_core_web_lg
  ```

---

## 🚀 Setup

1. **Clone the repo and install dependencies**
   ```bash
   git clone https://github.com/petasuhani/Graphitti.git
   cd Graphitti
   python -m venv venv
   source venv/bin/activate   # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```

2. **Configure environment variables**

   Create a `.env` file in the project root (this file is already git-ignored):
   ```env
   NEO4J_URI=your-neo4j-connection-uri
   NEO4J_USERNAME=your-username
   NEO4J_PASSWORD=your-password
   NEO4J_DATABASE=neo4j
   ```

3. **Start Ollama** in the background:
   ```bash
   ollama serve
   ```

4. **Run it**

   Streamlit app:
   ```bash
   streamlit run app.py
   ```

   Or the CLI version:
   ```bash
   python main.py
   ```

---

## 💻 Usage

### Streamlit app
1. Paste one URL per line into the text box (e.g. a few Wikipedia articles on a related topic)
2. Click **🚀 Run Pipeline** — Graphitti clears the previous graph, crawls, extracts entities/relationships, embeds text, builds the graph, and renders the visualization
3. Explore the interactive graph, then use the chat box to ask questions about what was crawled

### CLI
```bash
python main.py
```
Follow the prompts to enter the number of URLs and the URLs themselves. Once the pipeline finishes, an interactive question loop starts in the terminal (type `exit` to quit).

### Quick Neo4j connectivity check
```bash
python test_connection.py
```

---

## 🧩 Tech stack

| Layer | Tool |
|---|---|
| Crawling | `requests`, `BeautifulSoup` |
| NLP / entity & relation extraction | `spaCy` (`en_core_web_lg`) |
| Graph database | `Neo4j` |
| Graph visualization | `pyvis` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) via `langchain-huggingface` |
| Vector store | `ChromaDB` (via `langchain-chroma`) |
| Orchestration | `LangGraph` |
| LLM (classification, Cypher generation, answers) | `Llama 3` via `Ollama` (`langchain-ollama`) |
| Frontend | `Streamlit` |

---

## ⚠️ Known limitations

- Relationship extraction relies on spaCy's dependency parser, so it works best on clean, well-formed sentences (like Wikipedia prose) and can miss or misparse more complex sentence structures.
- Answer quality depends on the local Ollama/Llama 3 model — larger hosted models will generally produce better Cypher queries and answers.
- Running the pipeline clears the existing Neo4j graph, so each run currently represents one graph rather than incrementally growing one over time.

---

## 🔒 Important: rotate your Neo4j credentials

`config.py` currently has real Neo4j Aura credentials (URI, username, password) hardcoded and committed to this public repo. Since the repo is public, those credentials are exposed to anyone who looks. Before doing anything else:

1. In the Neo4j Aura console, reset/rotate the database password (and consider recreating the instance if you want to be extra safe).
2. Update `config.py` to read from environment variables instead, e.g.:
   ```python
   import os
   from dotenv import load_dotenv
   load_dotenv()

   NEO4J_URI = os.getenv("NEO4J_URI")
   NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
   NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
   ```
   This matches how `app.py`, `graph_retriever.py`, and `visualization.py` already load credentials, and keeps secrets out of git via the existing `.gitignore` entry for `.env`.
3. Consider scrubbing the old credentials from git history (e.g. with `git filter-repo` or BFG Repo-Cleaner) if you want them fully gone from the commit log — rotating the password is the critical step, but the old value will still be visible in past commits.
