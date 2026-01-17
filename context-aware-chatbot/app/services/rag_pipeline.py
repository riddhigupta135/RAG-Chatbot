"""
RAG (Retrieval-Augmented Generation) pipeline.
Combines retrieval from vector store with LLM generation.
"""

import json
import time
from typing import AsyncIterator, Optional

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from app.config import get_settings
from app.models.schemas import QueryResponse, SourceDocument
from app.services.vector_store import VectorStoreService
from app.utils.logging import get_logger

logger = get_logger(__name__)

# System prompt that enforces grounding in retrieved context (optimized for speed)
RAG_SYSTEM_PROMPT = """Answer questions using ONLY the provided context. Be concise and factual.

Rules:
1. Use ONLY information from the context
2. If context doesn't contain enough info, say "I don't have enough information in my knowledge base to answer this question fully"
3. Cite source(s) used
4. Keep answers brief and focused

Context:
{context}"""

RAG_USER_PROMPT = """Question: {question}

Answer concisely based on the context above. Cite sources when relevant."""


class RAGPipeline:
    """
    RAG pipeline that retrieves relevant documents and generates grounded responses.
    
    Features:
    - Retrieval from ChromaDB vector store
    - Context-aware prompt construction
    - LLM generation with Ollama
    - Source citation in responses
    - Streaming support
    """
    
    _instance: Optional["RAGPipeline"] = None
    
    def __new__(cls) -> "RAGPipeline":
        """Singleton pattern for resource efficiency."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        if self._initialized:
            return
            
        settings = get_settings()
        
        logger.info(
            "initializing_rag_pipeline",
            ollama_model=settings.ollama_model,
            ollama_url=settings.ollama_base_url,
        )
        
        # Initialize vector store
        self.vector_store = VectorStoreService()
        
        # Initialize LLM
        # Note: We'll use direct Ollama API calls instead of LangChain's Ollama wrapper
        # to have better control over timeouts
        self.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
            temperature=0.1,  # Low temperature for more factual responses
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ])
        
        self.settings = settings
        self._initialized = True
        
        logger.info("rag_pipeline_initialized")
        
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of relevant documents with scores.
        """
        top_k = top_k or self.settings.retrieval_top_k
        
        logger.info(
            "retrieving_documents",
            query=query[:100],
            top_k=top_k,
        )
        
        results = self.vector_store.search(query, top_k=top_k)
        
        logger.info(
            "retrieval_complete",
            num_results=len(results),
            scores=[r["score"] for r in results],
        )
        
        return results
        
    def _format_context(self, documents: list[dict]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: Retrieved documents with metadata.
            
        Returns:
            Formatted context string.
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc["metadata"].get("source", "Unknown source")
            title = doc["metadata"].get("title", "")
            content = doc["content"]
            
            header = f"[Source {i}: {title or source}]"
            context_parts.append(f"{header}\n{content}")
            
        return "\n\n---\n\n".join(context_parts)
        
    def _format_sources(self, documents: list[dict]) -> list[SourceDocument]:
        """
        Format documents into SourceDocument objects.
        
        Args:
            documents: Retrieved documents.
            
        Returns:
            List of SourceDocument objects.
        """
        sources = []
        
        for doc in documents:
            sources.append(SourceDocument(
                content=doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""),
                source=doc["metadata"].get("source", "Unknown"),
                title=doc["metadata"].get("title"),
                relevance_score=doc["score"],
                metadata=doc["metadata"],
            ))
            
        return sources
        
    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
    ) -> QueryResponse:
        """
        Process a RAG query end-to-end.
        
        Args:
            question: User's question.
            top_k: Number of documents to retrieve.
            include_sources: Whether to include sources in response.
            
        Returns:
            QueryResponse with answer and sources.
        """
        start_time = time.time()
        
        logger.info("processing_query", question=question[:100])
        
        # Retrieve relevant documents
        documents = self.retrieve(question, top_k)
        
        if not documents:
            logger.warning("no_documents_retrieved", question=question[:100])
            return QueryResponse(
                answer="I couldn't find any relevant information in my knowledge base to answer your question. Please try rephrasing or ask about a different topic.",
                sources=[],
                query_time_ms=(time.time() - start_time) * 1000,
            )
            
        # Format context from documents
        context = self._format_context(documents)
        
        # Generate response
        logger.info("generating_response", context_length=len(context))
        
        try:
            # Check if Ollama is accessible before making the request
            try:
                httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=5)
            except Exception as e:
                logger.error("ollama_not_accessible", error=str(e))
                return QueryResponse(
                    answer="Ollama is not running or not accessible. Please start Ollama first. On Windows, you can start it by running 'ollama serve' in a terminal.",
                    sources=self._format_sources(documents) if include_sources else [],
                    query_time_ms=(time.time() - start_time) * 1000,
                )
            
            prompt_value = self.prompt.format_messages(
                context=context,
                question=question,
            )
            
            # Convert messages to string for Ollama
            prompt_str = "\n".join([
                f"{msg.type}: {msg.content}" for msg in prompt_value
            ])
            
            logger.info("calling_ollama", prompt_length=len(prompt_str))
            
            # Call Ollama directly with proper timeout handling
            # Optimized for speed: reduced response length and faster inference
            try:
                timeout_config = httpx.Timeout(self.settings.ollama_timeout, connect=30.0)
                with httpx.Client(timeout=timeout_config) as client:
                    ollama_response = client.post(
                        f"{self.settings.ollama_base_url}/api/generate",
                        json={
                            "model": self.settings.ollama_model,
                            "prompt": prompt_str,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 300,  # Limit response length for faster generation
                                "num_ctx": 4096,  # Limit context window
                                "num_thread": 4,  # Use 4 threads for faster inference
                            }
                        }
                    )
                    ollama_response.raise_for_status()
                    result = ollama_response.json()
                    response = result.get("response", "")
                    if not response:
                        raise ValueError("Empty response from Ollama")
            except httpx.TimeoutException as e:
                logger.error("ollama_timeout", timeout=self.settings.ollama_timeout, error=str(e))
                return QueryResponse(
                    answer="The model is taking too long to respond. This often happens on the first request when the model needs to be loaded. Please wait a moment and try again.",
                    sources=self._format_sources(documents) if include_sources else [],
                    query_time_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                # Fallback to LangChain Ollama if direct API call fails
                logger.warning("direct_ollama_failed", error=str(e), error_type=type(e).__name__, fallback="langchain")
                try:
                    response = self.llm.invoke(prompt_str)
                except Exception as fallback_error:
                    logger.error("langchain_fallback_failed", error=str(fallback_error), error_type=type(fallback_error).__name__)
                    # Return error response instead of raising
                    return QueryResponse(
                        answer=f"Error generating response: {str(e)}. Fallback also failed: {str(fallback_error)}",
                        sources=self._format_sources(documents) if include_sources else [],
                        query_time_ms=(time.time() - start_time) * 1000,
                    )
            
            logger.info(
                "response_generated",
                response_length=len(response),
                query_time_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.error("generation_error", error=str(e))
            return QueryResponse(
                answer=f"I encountered an error while generating a response. Please try again. Error: {str(e)}",
                sources=self._format_sources(documents) if include_sources else [],
                query_time_ms=(time.time() - start_time) * 1000,
            )
            
        query_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=response,
            sources=self._format_sources(documents) if include_sources else [],
            query_time_ms=query_time,
        )
        
    async def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a RAG query response.
        
        Args:
            question: User's question.
            top_k: Number of documents to retrieve.
            
        Yields:
            Chunks of the response as they're generated.
        """
        logger.info("streaming_query", question=question[:100])
        
        # Retrieve relevant documents
        documents = self.retrieve(question, top_k)
        
        if not documents:
            yield "I couldn't find any relevant information in my knowledge base to answer your question."
            return
            
        # Format context
        context = self._format_context(documents)
        
        # Generate prompt
        prompt_value = self.prompt.format_messages(
            context=context,
            question=question,
        )
        
        prompt_str = "\n".join([
            f"{msg.type}: {msg.content}" for msg in prompt_value
        ])
        
        # Stream response
        try:
            for chunk in self.llm.stream(prompt_str):
                yield chunk
                
        except Exception as e:
            logger.error("streaming_error", error=str(e))
            yield f"\n\nError during generation: {str(e)}"
            
    def get_retrieved_sources(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> list[SourceDocument]:
        """
        Get just the retrieved sources without generating a response.
        Useful for showing sources alongside streaming responses.
        
        Args:
            question: User's question.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of source documents.
        """
        documents = self.retrieve(question, top_k)
        return self._format_sources(documents)
