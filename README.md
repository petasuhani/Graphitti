
# Graphitti — Graph-Native Web Intelligence

Graphitti turns web pages into a knowledge graph and answers natural-language questions using graph search and semantic search. It combines NLP, Neo4j, ChromaDB, and Llama 3 to retrieve relevant information and generate answers.

## Features

- **Web Crawling:** Extracts text from web pages using Requests and BeautifulSoup.
- **NLP Extraction:** Uses spaCy to identify entities and relationships.
- **Knowledge Graph:** Stores entities and relationships in Neo4j.
- **Graph Visualization:** Explores the knowledge graph using PyVis.
- **Semantic Search:** Uses sentence-transformers and ChromaDB to find relevant text.
- **Hybrid Retrieval:** Combines graph queries and vector search to answer questions.
- **Question Answering:** Uses Llama 3 through Ollama to generate answers from retrieved information.
- **User Interface:** Provides an interactive Streamlit app and a command-line interface.

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

### Prerequisites

- Python 3.10+
- A Neo4j database (local or Neo4j Aura)
- Ollama installed on your system

### 1. Clone the repository

```bash
git clone https://github.com/petasuhani/Graphitti.git
cd Graphitti
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Set up Ollama

Download the Llama 3 model:

```bash
ollama pull llama3
```

Make sure Ollama is running before starting the application.

### 4. Configure Neo4j

Create a `.env` file in the project directory and add your database credentials:

```env
NEO4J_URI=your-neo4j-uri
NEO4J_USERNAME=your-username
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

Replace the placeholder values with your actual credentials.

### 5. Run the application

Start the Streamlit interface:

```bash
streamlit run app.py
```

Or run the command-line version:

```bash
python main.py
```

## How It Works

1. Crawls web pages and extracts their text.
2. Uses spaCy to extract entities and relationships.
3. Stores the extracted information in Neo4j.
4. Converts text into embeddings and stores them in ChromaDB.
5. Classifies questions to determine whether graph search, vector search, or both are needed.
6. Retrieves relevant information and uses an LLM to generate an answer.

## Security

Keep your Neo4j credentials and API keys private. Store them in environment variables, add `.env` to `.gitignore`, and never commit secrets to a public repository.

## Limitations

- Relationship extraction depends on the quality and structure of the source text.
- Answer quality depends on the LLM and the information retrieved.
- Each pipeline run clears the existing graph, so previous graph data may be replaced.
```

**Before committing:** Make sure the setup instructions match your current code, `.env` is included in `.gitignore`, and any previously exposed Neo4j credentials have been rotated.
