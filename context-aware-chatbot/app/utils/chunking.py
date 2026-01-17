"""
Text chunking utilities for document processing.
Implements semantic chunking for better retrieval quality.
"""

from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: dict
    chunk_index: int


class SemanticChunker:
    """
    Semantic text chunker that respects document structure.
    
    Uses a combination of strategies:
    - Markdown header splitting for structured documents
    - Recursive character splitting with smart separators
    - Overlap for context preservation
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",    # Multiple newlines (section breaks)
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentence endings
                "? ",         # Question endings
                "! ",         # Exclamation endings
                "; ",         # Semicolons
                ", ",         # Commas
                " ",          # Words
                "",           # Characters
            ],
            length_function=len,
        )
        
        # Markdown header splitter for structured documents
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ],
            strip_headers=False,
        )
        
    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[TextChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text content to chunk.
            metadata: Base metadata to include with each chunk.
            
        Returns:
            List of TextChunk objects.
        """
        metadata = metadata or {}
        chunks = []
        
        logger.debug(
            "chunking_text",
            text_length=len(text),
            chunk_size=self.chunk_size,
        )
        
        # Check if text appears to be markdown
        is_markdown = text.strip().startswith("#") or "\n## " in text
        
        if is_markdown:
            # Use markdown-aware splitting
            md_chunks = self.md_splitter.split_text(text)
            
            # Further split large chunks
            for md_chunk in md_chunks:
                chunk_text = md_chunk.page_content
                chunk_metadata = {**metadata, **md_chunk.metadata}
                
                if len(chunk_text) > self.chunk_size:
                    # Split large chunks further
                    sub_chunks = self.recursive_splitter.split_text(chunk_text)
                    for sub_chunk in sub_chunks:
                        chunks.append(TextChunk(
                            content=sub_chunk,
                            metadata=chunk_metadata.copy(),
                            chunk_index=len(chunks),
                        ))
                else:
                    chunks.append(TextChunk(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        chunk_index=len(chunks),
                    ))
        else:
            # Use recursive splitting for plain text
            split_texts = self.recursive_splitter.split_text(text)
            
            for i, chunk_text in enumerate(split_texts):
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    chunk_index=i,
                ))
                
        # Add chunk-level metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata["chunk_index"] = chunk.chunk_index
            chunk.metadata["total_chunks"] = total_chunks
            
        logger.info(
            "chunking_complete",
            input_length=len(text),
            num_chunks=total_chunks,
            avg_chunk_size=len(text) // max(total_chunks, 1),
        )
        
        return chunks
        
    def chunk_documents(
        self,
        documents: list[dict],
    ) -> list[TextChunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents with 'content' and 'metadata' keys.
            
        Returns:
            List of TextChunk objects from all documents.
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            doc_chunks = self.chunk_text(content, metadata)
            all_chunks.extend(doc_chunks)
            
        logger.info(
            "batch_chunking_complete",
            num_documents=len(documents),
            total_chunks=len(all_chunks),
        )
        
        return all_chunks
