"""
Pydantic schemas for API requests and responses.
Provides type safety and automatic validation for all endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class IngestType(str, Enum):
    """Types of content that can be ingested."""
    URL = "url"
    FILE = "file"
    TEXT = "text"
    DIRECTORY = "directory"


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    
    type: IngestType = Field(
        description="Type of content to ingest"
    )
    source: str = Field(
        description="URL, file path, or raw text depending on type"
    )
    follow_links: bool = Field(
        default=False,
        description="For URL type: whether to follow and scrape linked pages"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to attach to ingested documents"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "url",
                    "source": "https://docs.company.com/handbook",
                    "follow_links": True,
                    "metadata": {"department": "HR"}
                },
                {
                    "type": "text",
                    "source": "Company policy states that...",
                    "metadata": {"category": "policy"}
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    
    success: bool = Field(description="Whether ingestion was successful")
    message: str = Field(description="Status message")
    documents_processed: int = Field(
        default=0,
        description="Number of documents processed"
    )
    chunks_created: int = Field(
        default=0,
        description="Number of chunks created and stored"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered during ingestion"
    )


class SourceDocument(BaseModel):
    """Represents a source document used to generate a response."""
    
    content: str = Field(description="Relevant content from the source")
    source: str = Field(description="Source URL or file path")
    title: Optional[str] = Field(
        default=None,
        description="Document title if available"
    )
    relevance_score: float = Field(
        description="Similarity score (0-1) indicating relevance"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the source"
    )


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    
    question: str = Field(
        min_length=1,
        description="The user's question"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of source documents to retrieve"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source documents in response"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the company's remote work policy?",
                    "top_k": 5,
                    "include_sources": True
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    
    answer: str = Field(description="Generated answer grounded in sources")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used to generate the answer"
    )
    query_time_ms: float = Field(
        description="Time taken to process the query in milliseconds"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the answer (0-1)"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(description="Health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the health check"
    )
    version: str = Field(description="Application version")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components"
    )


class CollectionStats(BaseModel):
    """Statistics about the vector collection."""
    
    collection_name: str = Field(description="Name of the collection")
    document_count: int = Field(description="Number of documents/chunks stored")
    embedding_dimension: int = Field(description="Dimension of embeddings")
