"""API routers for the RAG chatbot."""

from app.routers.ingest import router as ingest_router
from app.routers.query import router as query_router

__all__ = ["ingest_router", "query_router"]
