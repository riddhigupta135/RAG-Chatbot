"""
Vector store service using ChromaDB.
Handles embedding generation, storage, and similarity search.
"""

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Service for managing vector embeddings with ChromaDB.
    
    Features:
    - Local embedding generation using sentence-transformers
    - Persistent storage with ChromaDB
    - Similarity search with metadata filtering
    """
    
    _instance: Optional["VectorStoreService"] = None
    
    def __new__(cls) -> "VectorStoreService":
        """Singleton pattern to reuse embedding model and DB connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the vector store service."""
        if self._initialized:
            return
            
        settings = get_settings()
        
        logger.info(
            "initializing_vector_store",
            embedding_model=settings.embedding_model,
            persist_dir=settings.chroma_persist_dir,
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        self._initialized = True
        
        logger.info(
            "vector_store_initialized",
            collection_name=settings.chroma_collection_name,
            document_count=self.collection.count(),
            embedding_dimension=self.embedding_dimension,
        )
        
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        logger.debug("generating_embeddings", num_texts=len(texts))
        
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        return embeddings.tolist()
        
    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts.
            metadatas: List of metadata dicts for each document.
            ids: List of unique IDs for each document.
        """
        if not documents:
            logger.warning("no_documents_to_add")
            return
            
        logger.info(
            "adding_documents",
            num_documents=len(documents),
        )
        
        # Generate embeddings
        embeddings = self.generate_embeddings(documents)
        
        # Add to ChromaDB in batches to handle large datasets
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
            )
            
            logger.debug(
                "batch_added",
                batch_start=i,
                batch_end=batch_end,
            )
            
        logger.info(
            "documents_added",
            total_documents=len(documents),
            collection_count=self.collection.count(),
        )
        
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            where: Optional metadata filter.
            
        Returns:
            List of matching documents with scores.
        """
        logger.info(
            "searching_documents",
            query_length=len(query),
            top_k=top_k,
        )
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity score (cosine distance -> similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # Cosine distance to similarity
                
                formatted_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": similarity,
                })
                
        logger.info(
            "search_complete",
            num_results=len(formatted_results),
            top_score=formatted_results[0]["score"] if formatted_results else 0,
        )
        
        return formatted_results
        
    def delete_collection(self) -> None:
        """Delete all documents from the collection."""
        logger.warning("deleting_collection", collection_name=self.collection.name)
        
        settings = get_settings()
        self.chroma_client.delete_collection(settings.chroma_collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info("collection_deleted")
        
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "embedding_dimension": self.embedding_dimension,
        }
