"""
Configuration module for the RAG chatbot.
Loads settings from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings loaded from environment."""
    
    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"  # Faster, lighter model (3B parameters vs 7B+)
    ollama_timeout: int = 300  # Increased to 5 minutes for model loading
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # ChromaDB Configuration
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "company_docs"
    
    # Retrieval Settings
    retrieval_top_k: int = 3  # Reduced from 5 for faster processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid reloading env vars on every call.
    """
    return Settings()
