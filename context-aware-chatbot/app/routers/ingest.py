"""
Ingestion API endpoints.
Handles document ingestion from various sources.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import IngestRequest, IngestResponse
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStoreService
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    summary="Ingest documents",
    description="Ingest documents from URL, file, directory, or raw text into the vector store.",
)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """
    Ingest documents into the vector store.
    
    Supports multiple input types:
    - **url**: Scrape content from a URL (optionally follow links)
    - **file**: Load content from a local file
    - **directory**: Load all supported files from a directory
    - **text**: Ingest raw text content
    """
    logger.info(
        "ingest_request_received",
        type=request.type,
        source=request.source[:100] if request.source else None,
        follow_links=request.follow_links,
    )
    
    try:
        service = IngestionService()
        result = await service.ingest(
            ingest_type=request.type,
            source=request.source,
            follow_links=request.follow_links,
            metadata=request.metadata,
        )
        
        return IngestResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            errors=result.get("errors", []),
        )
        
    except Exception as e:
        logger.error("ingest_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post(
    "/refresh",
    response_model=IngestResponse,
    summary="Refresh vector store",
    description="Clear existing documents and re-ingest from source.",
)
async def refresh_documents(request: IngestRequest) -> IngestResponse:
    """
    Clear the vector store and re-ingest documents.
    
    This is useful when source documents have been updated and you want
    to ensure the vector store reflects the latest content.
    """
    logger.info("refresh_request_received")
    
    try:
        # Clear existing documents
        vector_store = VectorStoreService()
        vector_store.delete_collection()
        
        # Re-ingest
        service = IngestionService()
        result = await service.ingest(
            ingest_type=request.type,
            source=request.source,
            follow_links=request.follow_links,
            metadata=request.metadata,
        )
        
        return IngestResponse(
            success=result.get("success", False),
            message=f"Collection refreshed. {result.get('message', '')}",
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            errors=result.get("errors", []),
        )
        
    except Exception as e:
        logger.error("refresh_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Refresh failed: {str(e)}",
        )


@router.delete(
    "",
    summary="Clear vector store",
    description="Delete all documents from the vector store.",
)
async def clear_documents() -> JSONResponse:
    """
    Clear all documents from the vector store.
    
    Warning: This action cannot be undone.
    """
    logger.warning("clear_request_received")
    
    try:
        vector_store = VectorStoreService()
        vector_store.delete_collection()
        
        return JSONResponse(
            status_code=200,
            content={"message": "Vector store cleared successfully"},
        )
        
    except Exception as e:
        logger.error("clear_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Clear failed: {str(e)}",
        )


@router.get(
    "/stats",
    summary="Get ingestion statistics",
    description="Get statistics about the ingested documents.",
)
async def get_stats() -> dict:
    """Get statistics about the vector store."""
    try:
        vector_store = VectorStoreService()
        return vector_store.get_stats()
        
    except Exception as e:
        logger.error("stats_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}",
        )
