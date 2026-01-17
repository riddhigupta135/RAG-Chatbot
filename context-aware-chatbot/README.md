# Context-Aware RAG Chatbot

A Retrieval-Augmented Generation (RAG) system for internal employee queries. This chatbot retrieves information from internal company documentation and provides source-cited answers.

## Features

- **Semantic Search**: Intelligent document retrieval using vector embeddings
- **Local LLM**: Privacy-first approach using Ollama (Mistral/LLaMA)
- **Source Citations**: Every answer includes references to source documents
- **Web Scraping**: Automatically ingest content from internal websites
- **Chat Interface**: Beautiful Chainlit UI with streaming responses
- **REST API**: FastAPI backend for programmatic access
- **Observability**: Structured logging for monitoring and debugging

## Architecture

The system follows a layered architecture with clear separation of concerns:

**Frontend Layer:**
- **Web Interface**: A simple HTML/JavaScript frontend served directly by FastAPI, providing a user-friendly chat interface without requiring external UI frameworks.

**API Layer:**
- **FastAPI Backend**: RESTful API that handles all requests, manages authentication, and routes queries to the RAG pipeline. Provides endpoints for document ingestion and querying.

**Application Layer:**
- **RAG Pipeline**: The core orchestration component that coordinates retrieval and generation. It combines retrieved context from the vector store with the user's query and sends it to the LLM for answer generation.

**Data Layer:**
- **ChromaDB Vector Store**: Persistent vector database that stores document embeddings along with metadata. Enables fast similarity search for retrieving relevant context.
- **Sentence Transformers**: Local embedding model (all-MiniLM-L6-v2) that generates vector representations of text chunks for semantic similarity matching.

**LLM Layer:**
- **Ollama (Local LLM)**: Runs large language models locally for privacy and data control. The system uses llama3.2:3b by default for faster responses, but can be configured to use other models like Mistral or LLaMA.

**Processing Flow:**
1. User submits a question through the web interface or API
2. FastAPI receives the request and forwards it to the RAG Pipeline
3. RAG Pipeline queries ChromaDB using embeddings from Sentence Transformers to find relevant document chunks
4. Retrieved context is combined with the user's question in a structured prompt
5. The prompt is sent to Ollama (local LLM) to generate a grounded answer
6. The response, along with source citations, is returned to the user

**Ingestion Flow:**
1. Documents are loaded from URLs, files, or directories
2. Text is chunked semantically using LangChain's text splitters (respects document structure)
3. Each chunk is embedded using Sentence Transformers
4. Embeddings and metadata are stored in ChromaDB for later retrieval

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd context-aware-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy env.example .env  # Windows
# cp env.example .env  # Linux/Mac
```

### Start Ollama

```bash
# Pull the Mistral model (or your preferred model)
ollama pull mistral

# Ollama runs automatically as a service
# Verify it's running:
ollama list
```

### Ingest Sample Documents

```bash
# Start the API server
python -m uvicorn app.main:app --reload

# In another terminal, ingest sample documents
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"type": "directory", "source": "./data/sample_docs"}'
```

### Run the Chat Interface

```bash
# Start Chainlit
chainlit run chainlit_app.py
```

Open http://localhost:8000 for the chat interface.

## Usage

### API Endpoints

#### Ingest Documents

```bash
# Ingest from URL
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "url",
    "source": "https://docs.company.com",
    "follow_links": true
  }'

# Ingest from file
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "file",
    "source": "./data/policy.md"
  }'

# Ingest raw text
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text",
    "source": "Company policy states that...",
    "metadata": {"category": "policy"}
  }'
```

#### Query the RAG System

```bash
# Standard query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the remote work policy?",
    "top_k": 5,
    "include_sources": true
  }'

# Streaming query
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I request time off?"}'
```

### Chat Interface

The Chainlit interface provides:
- Real-time streaming responses
- Source document display
- Configurable retrieval settings
- Session-based conversation history

## Configuration

Edit `.env` to customize:

```env
# LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Retrieval Settings
RETRIEVAL_TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Project Structure

```
context-aware-chatbot/
  app/
    __init__.py
    config.py              
    main.py                
    models/
      __init__.py
      schemas.py         
    routers/
      __init__.py
      ingest.py          
      query.py          
    services/
      __init__.py
      ingestion.py       
      rag_pipeline.py    
      vector_store.py    
    utils/
      __init__.py
      chunking.py        
      logging.py        
      scraper.py         
  chainlit_app.py           
  data/
    sample_docs/           
  chroma_db/                 
  requirements.txt
  env.example
  README.md
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Logging

The application uses structured logging with `structlog`. Logs include:
- Ingestion events
- Retrieval results
- LLM responses
- Error tracking

View logs in JSON format for easy parsing by log aggregators.

### Adding New Document Sources

1. Implement a new method in `app/services/ingestion.py`
2. Add corresponding `IngestType` to `app/models/schemas.py`
3. Update the router in `app/routers/ingest.py`

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
# Windows: Restart from system tray
# Linux: systemctl restart ollama
```

### ChromaDB Issues

```bash
# Clear vector store
curl -X DELETE http://localhost:8000/ingest

# Re-ingest documents
curl -X POST http://localhost:8000/ingest/refresh \
  -H "Content-Type: application/json" \
  -d '{"type": "directory", "source": "./data/sample_docs"}'
```

### Memory Issues

If running out of memory with large documents:
1. Reduce `CHUNK_SIZE` in `.env`
2. Process documents in smaller batches
3. Use a smaller embedding model

## Security Considerations

- This system is designed for **internal use only**
- Implement authentication before deploying
- Restrict CORS origins in production
- Use environment variables for sensitive configuration
- Audit and rotate API keys regularly
