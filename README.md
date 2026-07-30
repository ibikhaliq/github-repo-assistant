# GitHub Repository Assistant

An AI-powered repository intelligence platform that enables developers to quickly understand unfamiliar GitHub repositories through semantic search and natural language question answering.

The application clones public GitHub repositories, generates AI-powered file summaries, creates semantic embeddings using an open-source embedding model, and answers repository-specific questions by retrieving the most relevant code and documentation.

---

## Features

- Clone and analyze any public GitHub repository
- Automatically summarize source files using an LLM
- Generate semantic embeddings with **Ollama** and **Nomic Embed Text**
- Perform semantic retrieval over repository contents
- Answer repository-specific questions using retrieval-augmented generation (RAG)
- Commit-aware indexing that avoids reprocessing unchanged repositories
- Batch embedding generation for faster indexing
- Dockerized deployment
- Responsive web interface built with FastAPI

---

## System Architecture

```text
                 GitHub Repository
                        │
                        ▼
                Repository Cloning
                        │
                        ▼
              AI File Summarization
                        │
                        ▼
      Semantic Embedding Generation
          (Ollama + Nomic Embed)
                        │
                        ▼
           Local Vector Repository
                        │
                        ▼
          Hybrid Semantic Retrieval
                        │
                        ▼
         SpiderAI Large Language Model
                        │
                        ▼
               Repository Answers
```

---

## Technology Stack

### Backend

- Python
- FastAPI
- Jinja2
- GitPython

### Artificial Intelligence

- SpiderAI (LLM reasoning)
- Ollama
- Nomic Embed Text
- Retrieval-Augmented Generation (RAG)

### Infrastructure

- Docker
- Git
- GitHub

---

## Key Capabilities

### Repository Cloning

Clone and locally index any public GitHub repository.

### AI Summarization

Each supported source file is summarized using a large language model to improve retrieval quality while reducing prompt size.

### Semantic Search

Repository summaries are converted into vector embeddings using the open-source **Nomic Embed Text** model served through **Ollama**.

### Intelligent Question Answering

Natural language questions are answered using:

- semantic retrieval
- repository summaries
- source code context
- LLM reasoning

---

## Performance Optimizations

The project includes several optimizations to improve indexing performance:

- Commit-aware repository caching
- Parallel file summarization
- Batched embedding generation
- Cached repository indexing
- Incremental repository updates

These optimizations significantly reduce repeated indexing time for repositories that have not changed.

---

## Project Structure

```
backend/
│
├── static/
├── templates/
│
├── main.py
├── github_service.py
├── repository_service.py
├── summary_service.py
├── embedding_service.py
├── semantic_retrieval_service.py
├── hybrid_retrieval_service.py
├── retrieval_service.py
├── vector_store.py
├── ai_service.py
├── prompts.py
├── models.py
│
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

---

## Running Locally

### Clone the repository

```bash
git clone https://github.com/<your-username>/github-repo-assistant.git
cd github-repo-assistant/backend
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama serve
```

### Download the embedding model

```bash
ollama pull nomic-embed-text
```

### Configure environment variables

Create a `.env` file containing:

```env
SPIDERAI_BASE_URL=
SPIDERAI_API_KEY=
SPIDERAI_MODEL=

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Start the application

```bash
uvicorn main:app --reload
```

Open:

```
http://localhost:8000
```

---

## Docker

Build

```bash
docker build -t github-repo-assistant .
```

Run

```bash
docker run \
    --env-file .env \
    -p 8000:8000 \
    github-repo-assistant
```

---

## Example Questions

- Where is authentication implemented?
- How does the application handle user input?
- Explain the project architecture.
- Where is the database initialized?
- Which files contain API routes?
- How is caching implemented?
- What technologies does this repository use?

---

## Future Improvements

- Persistent vector database
- GitHub authentication for private repositories
- Multi-repository search
- Repository comparison
- Streaming LLM responses
- Conversation memory
- Deployment on AWS

---

## License

MIT License
