# Sherlock RAG

A local Retrieval-Augmented Generation (RAG) system that lets you ask questions about *The Adventures of Sherlock Holmes* through a web UI.

**Stack:** LM Studio (Llama 3.1 8B) · LangChain · ChromaDB · FastAPI · Blazor Server (.NET 10)

## Architecture

```
┌───────────────┐       HTTP POST       ┌───────────────┐      OpenAI-compat      ┌─────────────┐
│  Blazor UI    │ ───────/query───────▸ │  FastAPI      │ ─────────────────────▸  │  LM Studio  │
│  (localhost:  │                       │  (localhost:  │                         │  (localhost:│
│   5165)       │ ◂──── JSON ─────────  │   8000)       │ ◂────────────────────   │   1234)     │
└───────────────┘                       └──────┬────────┘                         └─────────────┘
                                               │
                                               │ similarity search
                                               ▼
                                       ┌───────────────┐
                                       │   ChromaDB    │
                                       │ (vector store)│
                                       └───────────────┘
```

### How it works

1. **Ingestion (one-time).** The book text is split into overlapping chunks (~1 000 characters each). Each chunk is converted into a numeric vector (embedding) by the **all-MiniLM-L6-v2** model and stored in a **ChromaDB** vector database on disk. This step runs locally — the embedding model is downloaded automatically by the `sentence-transformers` library on first run (~80 MB) and requires no API keys.

2. **Query flow.** When you ask a question:
   - The question text is embedded with the same **all-MiniLM-L6-v2** model.
   - ChromaDB finds the most similar chunks using cosine similarity (semantic search).
   - The retrieved chunks are combined into a prompt and sent, together with your question, to the **LLM** running in LM Studio via its OpenAI-compatible API.
   - The LLM generates an answer grounded in the retrieved context.

3. **Frontend.** Blazor Server renders a simple form, sends your question to the FastAPI backend over HTTP, and displays the answer.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [LM Studio](https://lmstudio.ai/) | Latest | Run the LLM locally |
| [Python](https://www.python.org/downloads/) | 3.11+ | Backend |
| [.NET SDK](https://dotnet.microsoft.com/download) | 10.0 | Frontend |

## Quick Start

### 1. Download the model in LM Studio

Open LM Studio and download **Meta-Llama-3.1-8B-Instruct** (Q4_K_M quantization recommended for ≤16 GB RAM).
Start the local server — it should be available at `http://localhost:1234`.

### 2. Set up the Python backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy and edit the config
cp .env.example .env
```

### 3. Build the vector store

This reads the book from `data/`, splits it into chunks, generates embeddings with **all-MiniLM-L6-v2**, and saves everything into a ChromaDB database at `backend/chroma_db/`.

The embedding model (~80 MB) is downloaded automatically on first run. You only need to run this step once (or again after changing the source text).

```bash
python rag_ingest.py
```

### 4. Start the API server

```bash
python rag_api.py
```

The API will be available at `http://localhost:8000`.
You can test it directly:

**PowerShell:**

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query -ContentType "application/json" -Body '{"question": "Where does Sherlock Holmes live?"}'
```

**bash / macOS / Linux:**

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "Where does Sherlock Holmes live?"}'
```

### 5. Start the web UI

Open a **new terminal**:

```bash
cd frontend/SherlockUI
dotnet run
```

Open `http://localhost:5165` in your browser.

## Project Structure

```
RAG/
├── backend/
│   ├── config.py           # Settings loaded from .env
│   ├── rag_chain.py        # Embedding function, vector store, QA chain (LCEL)
│   ├── rag_ingest.py       # One-time: split book → embed → ChromaDB
│   ├── rag_api.py          # FastAPI server (/query endpoint)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── SherlockUI/         # Blazor Server web app
├── data/
│   └── The_Adventures_of_Sherlock_Holmes.txt
└── README.md
```

## Configuration

All backend settings are in `backend/.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) for splitting |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `DATA_DIR` | `../data` | Path to source documents |
| `CHROMA_DIR` | `./chroma_db` | Path to ChromaDB storage |

## Example Questions

- Where does Sherlock Holmes live?
- What was the connection between Irene Adler and the King of Bohemia?
- Describe the appearance of the visitor in "The Red-Headed League".
- How did Holmes solve the mystery in "A Case of Identity"?

## License

This is a personal learning project. The book text is in the public domain.
