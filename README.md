
# Graphitti — Graph-Native Web Intelligence

Graphitti converts web pages into a knowledge graph and answers natural-language questions using graph search and semantic search. It combines NLP, Neo4j, ChromaDB, and Llama 3 to retrieve relevant information and generate answers.

## Features

- Web crawling using Requests and BeautifulSoup
- Entity and relationship extraction using spaCy
- Knowledge graph construction using Neo4j
- Interactive graph visualization using PyVis
- Semantic search using ChromaDB and sentence-transformers
- Hybrid retrieval and question answering using Llama 3
- Interactive Streamlit interface and CLI support

## Architecture

```text
          Web URLs
              |
              v
       crawler.py
    (Extracts web text)
              |
              v
          nlp.py
 (Extracts entities and relations)
              |
              v
      graph_pipeline.py
              |
              v
          Neo4j
     (Knowledge Graph)

  Web Text ---> embeddings.py ---> chroma.py
                                      |
                                      v
                                  ChromaDB
                               (Vector Search)

          User Question
                |
                v
      query_classifier.py
                |
                v
       hybrid_retriever.py
          /           \
         v             v
graph_retriever.py  vector_retriever.py
         \             /
          v           v
       Retrieved Context
                |
                v
      answer_generator.py
                |
                v
          Final Answer
```

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `main.py` | Command-line interface |
| `crawler.py` | Fetches and cleans web page content |
| `nlp.py` | Extracts entities and relationships |
| `graph_pipeline.py` | Stores extracted information in Neo4j |
| `visualization.py` | Creates interactive graph visualizations |
| `embeddings.py` | Generates text embeddings |
| `chroma.py` | Manages the ChromaDB vector store |
| `query_classifier.py` | Classifies questions for retrieval |
| `graph_retriever.py` | Retrieves information from Neo4j |
| `vector_retriever.py` | Retrieves relevant text using vector search |
| `hybrid_retriever.py` | Combines graph and vector retrieval |
| `answer_generator.py` | Generates answers from retrieved context |
| `langgraph_pipeline.py` | Orchestrates retrieval and answer generation |
| `config.py` | Manages configuration settings |

## Tech Stack

- **Language:** Python
- **Web Crawling:** Requests, BeautifulSoup
- **NLP:** spaCy
- **Graph Database:** Neo4j
- **Vector Store:** ChromaDB
- **Embeddings:** Sentence Transformers
- **LLM:** Llama 3 via Ollama
- **Orchestration:** LangGraph
- **Visualization:** PyVis
- **Frontend:** Streamlit

## Setup

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download the spaCy model using `python -m spacy download en_core_web_lg`.
4. Install Ollama and download Llama 3 using `ollama pull llama3`.
5. Configure Neo4j credentials using environment variables in a `.env` file.
6. Run `streamlit run app.py` to start the application.

Alternatively, run `python main.py` to use the command-line interface.

## SECURITY

Keep your Neo4j credentials and API keys private. Store them in environment variables, add `.env` to `.gitignore`, and never commit secrets to a public repository.


**Before publishing:**
- Make sure the file names in the architecture and project structure match your actual repository.
- Confirm that the `.env` file is ignored by Git.
- If real Neo4j credentials were committed publicly, rotate them immediately.

The architecture diagram is written in a simple text format, so GitHub will display it directly without requiring an image.
