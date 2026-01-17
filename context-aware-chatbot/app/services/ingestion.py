"""
Document ingestion service.
Handles loading, processing, and storing documents from various sources.
"""

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models.schemas import IngestType
from app.services.vector_store import VectorStoreService
from app.utils.chunking import SemanticChunker
from app.utils.logging import get_logger
from app.utils.scraper import WebScraper

logger = get_logger(__name__)


class IngestionService:
    """
    Service for ingesting documents into the vector store.
    
    Supports multiple input types:
    - URLs (with optional link following)
    - Local files (.txt, .md, .html)
    - Directories of documents
    - Raw text content
    """
    
    def __init__(self):
        """Initialize the ingestion service."""
        self.vector_store = VectorStoreService()
        self.chunker = SemanticChunker()
        self.scraper = WebScraper()
        
    async def ingest_url(
        self,
        url: str,
        follow_links: bool = False,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest content from a URL.
        
        Args:
            url: URL to scrape.
            follow_links: Whether to follow internal links.
            metadata: Additional metadata to attach.
            
        Returns:
            Ingestion result with stats.
        """
        logger.info(
            "ingesting_url",
            url=url,
            follow_links=follow_links,
        )
        
        metadata = metadata or {}
        
        # Scrape the URL(s)
        self.scraper = WebScraper()  # Reset visited URLs
        documents = await self.scraper.scrape_site(url, follow_links=follow_links)
        
        if not documents:
            return {
                "success": False,
                "message": "No content could be extracted from the URL",
                "documents_processed": 0,
                "chunks_created": 0,
            }
            
        # Process and store documents
        return self._process_documents(
            documents=[
                {
                    "content": doc.content,
                    "metadata": {**doc.metadata, **metadata},
                }
                for doc in documents
            ]
        )
        
    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest content from a local file.
        
        Args:
            file_path: Path to the file.
            metadata: Additional metadata to attach.
            
        Returns:
            Ingestion result with stats.
        """
        logger.info("ingesting_file", file_path=file_path)
        
        path = Path(file_path)
        
        if not path.exists():
            return {
                "success": False,
                "message": f"File not found: {file_path}",
                "documents_processed": 0,
                "chunks_created": 0,
            }
            
        metadata = metadata or {}
        metadata["source"] = str(path.absolute())
        metadata["filename"] = path.name
        metadata["type"] = "file"
        
        # Read file content
        content = self._read_file(path)
        
        if not content:
            return {
                "success": False,
                "message": f"Could not read file: {file_path}",
                "documents_processed": 0,
                "chunks_created": 0,
            }
            
        return self._process_documents([
            {"content": content, "metadata": metadata}
        ])
        
    def ingest_directory(
        self,
        directory_path: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory_path: Path to the directory.
            metadata: Additional metadata to attach to all documents.
            
        Returns:
            Ingestion result with stats.
        """
        logger.info("ingesting_directory", directory_path=directory_path)
        
        dir_path = Path(directory_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return {
                "success": False,
                "message": f"Directory not found: {directory_path}",
                "documents_processed": 0,
                "chunks_created": 0,
            }
            
        metadata = metadata or {}
        documents = []
        supported_extensions = {".txt", ".md", ".html", ".htm"}
        
        # Find all supported files
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                content = self._read_file(file_path)
                
                if content:
                    doc_metadata = {
                        **metadata,
                        "source": str(file_path.absolute()),
                        "filename": file_path.name,
                        "type": "file",
                    }
                    documents.append({
                        "content": content,
                        "metadata": doc_metadata,
                    })
                    
        if not documents:
            return {
                "success": False,
                "message": "No supported files found in directory",
                "documents_processed": 0,
                "chunks_created": 0,
            }
            
        return self._process_documents(documents)
        
    def ingest_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest raw text content.
        
        Args:
            text: Raw text content.
            metadata: Metadata to attach.
            
        Returns:
            Ingestion result with stats.
        """
        logger.info("ingesting_text", text_length=len(text))
        
        metadata = metadata or {}
        metadata["type"] = "text"
        metadata["source"] = "direct_input"
        
        return self._process_documents([
            {"content": text, "metadata": metadata}
        ])
        
    def _read_file(self, path: Path) -> Optional[str]:
        """Read content from a file."""
        try:
            # Handle HTML files specially
            if path.suffix.lower() in {".html", ".htm"}:
                from bs4 import BeautifulSoup
                
                with open(path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "lxml")
                    
                # Remove script and style elements
                for element in soup(["script", "style"]):
                    element.decompose()
                    
                return soup.get_text(separator="\n", strip=True)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
                    
        except Exception as e:
            logger.error("file_read_error", path=str(path), error=str(e))
            return None
            
    def _process_documents(self, documents: list[dict]) -> dict:
        """
        Process and store documents in the vector store.
        
        Args:
            documents: List of documents with content and metadata.
            
        Returns:
            Processing result with stats.
        """
        logger.info("processing_documents", num_documents=len(documents))
        
        # Chunk all documents
        all_chunks = self.chunker.chunk_documents(documents)
        
        if not all_chunks:
            return {
                "success": False,
                "message": "No chunks created from documents",
                "documents_processed": len(documents),
                "chunks_created": 0,
            }
            
        # Generate unique IDs for each chunk
        chunk_texts = []
        chunk_metadatas = []
        chunk_ids = []
        
        for chunk in all_chunks:
            # Create deterministic ID from content hash
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:12]
            chunk_id = f"{chunk.metadata.get('source', 'unknown')}_{chunk.chunk_index}_{content_hash}"
            
            chunk_texts.append(chunk.content)
            chunk_metadatas.append(chunk.metadata)
            chunk_ids.append(chunk_id)
            
        # Store in vector store
        try:
            self.vector_store.add_documents(
                documents=chunk_texts,
                metadatas=chunk_metadatas,
                ids=chunk_ids,
            )
            
            logger.info(
                "ingestion_complete",
                documents_processed=len(documents),
                chunks_created=len(all_chunks),
            )
            
            return {
                "success": True,
                "message": "Documents ingested successfully",
                "documents_processed": len(documents),
                "chunks_created": len(all_chunks),
            }
            
        except Exception as e:
            logger.error("ingestion_error", error=str(e))
            return {
                "success": False,
                "message": f"Error storing documents: {str(e)}",
                "documents_processed": len(documents),
                "chunks_created": 0,
                "errors": [str(e)],
            }
            
    async def ingest(
        self,
        ingest_type: IngestType,
        source: str,
        follow_links: bool = False,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Main ingestion method that routes to appropriate handler.
        
        Args:
            ingest_type: Type of content to ingest.
            source: Source URL, path, or text.
            follow_links: For URLs, whether to follow links.
            metadata: Additional metadata.
            
        Returns:
            Ingestion result.
        """
        if ingest_type == IngestType.URL:
            return await self.ingest_url(source, follow_links, metadata)
        elif ingest_type == IngestType.FILE:
            return self.ingest_file(source, metadata)
        elif ingest_type == IngestType.DIRECTORY:
            return self.ingest_directory(source, metadata)
        elif ingest_type == IngestType.TEXT:
            return self.ingest_text(source, metadata)
        else:
            return {
                "success": False,
                "message": f"Unknown ingest type: {ingest_type}",
                "documents_processed": 0,
                "chunks_created": 0,
            }
