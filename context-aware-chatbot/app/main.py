"""
FastAPI application entry point.
Sets up the API server with all routes and middleware.
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

from app import __version__
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.routers import ingest_router, query_router
from app.utils.logging import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs startup and shutdown logic.
    """
    # Startup
    logger.info("application_starting", version=__version__)
    
    # Pre-initialize services to fail fast if there are issues
    try:
        from app.services.vector_store import VectorStoreService
        from app.services.rag_pipeline import RAGPipeline
        
        # Initialize vector store
        vector_store = VectorStoreService()
        logger.info(
            "vector_store_ready",
            document_count=vector_store.collection.count(),
        )
        
        # Initialize RAG pipeline
        # Note: This will fail if Ollama is not running
        # We'll handle this gracefully
        try:
            rag = RAGPipeline()
            logger.info("rag_pipeline_ready")
        except Exception as e:
            logger.warning(
                "rag_pipeline_init_warning",
                error=str(e),
                message="RAG pipeline will initialize on first query",
            )
            
    except Exception as e:
        logger.error("startup_error", error=str(e))
        raise
        
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI application
app = FastAPI(
    title="RAG Chatbot API",
    description="""
    A Retrieval-Augmented Generation (RAG) chatbot API for internal employee queries.
    
    ## Features
    
    - **Document Ingestion**: Ingest content from URLs, files, or raw text
    - **Semantic Search**: Find relevant documents using vector similarity
    - **Grounded Responses**: Generate answers based only on retrieved sources
    - **Source Citations**: Every response includes references to source documents
    
    ## Quick Start
    
    1. **Ingest documents**: POST to `/ingest` with your content
    2. **Query**: POST to `/query` with your question
    3. **Get streaming responses**: POST to `/query/stream`
    
    ## Note
    
    This API requires Ollama to be running locally with a compatible model.
    """,
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest_router)
app.include_router(query_router)


@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Chat Interface",
    description="Web interface for the RAG chatbot.",
)
async def root():
    """Serve the chat interface."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chatbot - Company Knowledge Assistant</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .input-group {
            margin-bottom: 20px;
        }
        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-right: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .response {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            display: none;
        }
        .response.show {
            display: block;
        }
        .answer {
            margin-bottom: 20px;
            line-height: 1.8;
            color: #333;
            font-size: 16px;
        }
        .sources {
            margin-top: 25px;
            padding-top: 25px;
            border-top: 2px solid #e0e0e0;
        }
        .sources h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .source {
            background: white;
            padding: 15px;
            margin-bottom: 12px;
            border-radius: 8px;
            border-left: 3px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .source-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }
        .source-content {
            color: #666;
            font-size: 14px;
            margin-top: 8px;
            line-height: 1.6;
        }
        .source-score {
            color: #999;
            font-size: 12px;
            margin-top: 8px;
        }
        .loading {
            color: #667eea;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        .error {
            color: #d32f2f;
            background: #ffebee;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #d32f2f;
        }
        .query-time {
            margin-top: 15px;
            color: #999;
            font-size: 12px;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Chatbot</h1>
        <p class="subtitle">Ask questions about your company documentation</p>
        
        <div class="input-group">
            <input type="text" id="question" placeholder="e.g., What is the remote work policy? How do I request time off?" 
                   onkeypress="if(event.key === 'Enter' && !document.getElementById('askBtn').disabled) askQuestion()">
        </div>
        
        <button id="askBtn" onclick="askQuestion()">Ask Question</button>
        <button onclick="clearResponse()">Clear</button>
        
        <div id="response" class="response"></div>
    </div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('Please enter a question!');
                return;
            }
            
            const responseDiv = document.getElementById('response');
            const askBtn = document.getElementById('askBtn');
            
            responseDiv.className = 'response show';
            responseDiv.innerHTML = '<div class="loading">⏳ Processing your question... This may take 10-30 seconds.</div>';
            askBtn.disabled = true;
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question,
                        top_k: 5,
                        include_sources: true
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                let html = '<div class="response show">';
                html += '<h3>Answer:</h3>';
                html += '<div class="answer">' + data.answer.replace(/\\n/g, '<br>') + '</div>';
                
                if (data.sources && data.sources.length > 0) {
                    html += '<div class="sources"><h3>📚 Sources (' + data.sources.length + '):</h3>';
                    data.sources.forEach((source, index) => {
                        const sourceName = source.title || source.source.split(/[/\\\\]/).pop();
                        html += '<div class="source">';
                        html += '<div class="source-title">Source ' + (index + 1) + ': ' + sourceName + '</div>';
                        html += '<div class="source-content">' + source.content.substring(0, 250) + (source.content.length > 250 ? '...' : '') + '</div>';
                        html += '<div class="source-score">Relevance: ' + (source.relevance_score * 100).toFixed(1) + '%</div>';
                        html += '</div>';
                    });
                    html += '</div>';
                }
                
                if (data.query_time_ms) {
                    html += '<div class="query-time">Query processed in ' + (data.query_time_ms / 1000).toFixed(2) + ' seconds</div>';
                }
                
                html += '</div>';
                
                responseDiv.innerHTML = html;
                
            } catch (error) {
                responseDiv.innerHTML = '<div class="error"><strong>Error:</strong> ' + error.message + '<br><br>Make sure the API server is running and Ollama is available.</div>';
            } finally {
                askBtn.disabled = false;
            }
        }
        
        function clearResponse() {
            const responseDiv = document.getElementById('response');
            responseDiv.className = 'response';
            responseDiv.innerHTML = '';
            document.getElementById('question').value = '';
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of the API and its components.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the status of all components:
    - API: Always healthy if this endpoint responds
    - Vector Store: Checks ChromaDB connection
    - LLM: Checks Ollama availability
    """
    components = {"api": "healthy"}
    overall_status = "healthy"
    
    # Check vector store
    try:
        from app.services.vector_store import VectorStoreService
        vs = VectorStoreService()
        vs.collection.count()
        components["vector_store"] = "healthy"
    except Exception as e:
        components["vector_store"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
        
    # Check LLM
    try:
        import httpx
        settings = get_settings()
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            components["llm"] = "healthy"
        else:
            components["llm"] = f"unhealthy: status {response.status_code}"
            overall_status = "degraded"
    except Exception as e:
        components["llm"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
        
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=__version__,
        components=components,
    )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
