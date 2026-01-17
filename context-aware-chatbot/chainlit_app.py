"""
Chainlit chat interface for the RAG chatbot.
Provides a beautiful, interactive chat UI with streaming responses.
"""

import chainlit as cl
from chainlit.input_widget import Select, Slider

from app.config import get_settings
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStoreService
from app.utils.logging import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Initialize services
settings = get_settings()


@cl.on_chat_start
async def on_chat_start():
    """
    Called when a new chat session starts.
    Initialize the session and show welcome message.
    """
    logger.info("chat_session_started")
    
    # Initialize RAG pipeline for this session
    try:
        rag_pipeline = RAGPipeline()
        cl.user_session.set("rag_pipeline", rag_pipeline)
        
        # Get collection stats
        vector_store = VectorStoreService()
        stats = vector_store.get_stats()
        doc_count = stats.get("document_count", 0)
        
        # Settings for the session
        chat_settings = await cl.ChatSettings(
            [
                Slider(
                    id="top_k",
                    label="Number of sources to retrieve",
                    initial=settings.retrieval_top_k,
                    min=1,
                    max=10,
                    step=1,
                ),
                Select(
                    id="show_sources",
                    label="Show source documents",
                    values=["Yes", "No"],
                    initial_value="Yes",
                ),
            ]
        ).send()
        
        # Store settings
        cl.user_session.set("top_k", settings.retrieval_top_k)
        cl.user_session.set("show_sources", True)
        
        # Send welcome message
        welcome_message = f"""👋 **Welcome to the Company Knowledge Assistant!**

I'm here to help you find information from our internal documentation. I have access to **{doc_count}** indexed document chunks.

**How to use me:**
- Ask any question about company policies, procedures, or documentation
- I'll search our knowledge base and provide answers with source citations
- Use the settings (⚙️) to customize how many sources I retrieve

**Examples of questions you can ask:**
- "What is our remote work policy?"
- "How do I request time off?"
- "What are the security guidelines for handling customer data?"

Let me know how I can help! 🚀"""

        await cl.Message(content=welcome_message).send()
        
    except Exception as e:
        logger.error("session_init_error", error=str(e))
        await cl.Message(
            content=f"⚠️ Error initializing the assistant: {str(e)}\n\nPlease ensure Ollama is running with the '{settings.ollama_model}' model."
        ).send()


@cl.on_settings_update
async def on_settings_update(settings_dict: dict):
    """
    Called when chat settings are updated.
    Update session variables based on new settings.
    """
    logger.info("settings_updated", settings=settings_dict)
    
    if "top_k" in settings_dict:
        cl.user_session.set("top_k", int(settings_dict["top_k"]))
        
    if "show_sources" in settings_dict:
        cl.user_session.set("show_sources", settings_dict["show_sources"] == "Yes")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Called when the user sends a message.
    Process the query and stream the response.
    """
    question = message.content.strip()
    
    if not question:
        await cl.Message(content="Please enter a question.").send()
        return
        
    logger.info("user_message_received", question=question[:100])
    
    # Get session settings
    rag_pipeline: RAGPipeline = cl.user_session.get("rag_pipeline")
    top_k = cl.user_session.get("top_k", settings.retrieval_top_k)
    show_sources = cl.user_session.get("show_sources", True)
    
    if not rag_pipeline:
        await cl.Message(
            content="⚠️ Session not initialized. Please refresh the page."
        ).send()
        return
    
    # Create a message for streaming the response
    response_message = cl.Message(content="")
    await response_message.send()
    
    try:
        # Get sources first (for display alongside response)
        sources = rag_pipeline.get_retrieved_sources(question, top_k)
        
        if not sources:
            await response_message.stream_token(
                "I couldn't find any relevant information in my knowledge base to answer your question. "
                "Please try rephrasing or ask about a different topic."
            )
            await response_message.update()
            return
            
        # Stream the response
        full_response = ""
        async for chunk in rag_pipeline.stream_query(question, top_k):
            full_response += chunk
            await response_message.stream_token(chunk)
            
        await response_message.update()
        
        # Show sources if enabled
        if show_sources and sources:
            sources_text = "\n\n---\n\n📚 **Sources:**\n\n"
            
            for i, source in enumerate(sources, 1):
                title = source.title or "Untitled"
                score = source.relevance_score
                source_url = source.source
                
                # Truncate content for display
                content_preview = source.content[:200] + "..." if len(source.content) > 200 else source.content
                
                sources_text += f"""**{i}. {title}** (relevance: {score:.2%})
> {content_preview}
> *Source: {source_url}*

"""
            
            # Create a separate message for sources
            await cl.Message(content=sources_text).send()
            
        logger.info(
            "response_sent",
            question=question[:100],
            response_length=len(full_response),
            num_sources=len(sources),
        )
        
    except Exception as e:
        logger.error("message_processing_error", error=str(e))
        await response_message.stream_token(
            f"\n\n⚠️ An error occurred while processing your question: {str(e)}"
        )
        await response_message.update()


@cl.on_chat_end
async def on_chat_end():
    """Called when the chat session ends."""
    logger.info("chat_session_ended")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    Optional: Basic authentication for the chat interface.
    
    In production, integrate with your company's SSO/LDAP.
    For now, this is disabled but can be enabled by uncommenting
    the decorator below.
    
    To enable, add these to your .env:
    CHAINLIT_AUTH_SECRET=your-secret-key
    """
    # Example: Simple authentication (replace with your auth logic)
    # if username == "admin" and password == "password":
    #     return cl.User(identifier=username, metadata={"role": "admin"})
    # return None
    pass


# Custom actions for common tasks
@cl.action_callback("clear_chat")
async def on_clear_chat(action: cl.Action):
    """Clear the chat history."""
    await cl.Message(content="Chat cleared. How can I help you?").send()


@cl.action_callback("show_stats")
async def on_show_stats(action: cl.Action):
    """Show vector store statistics."""
    try:
        vector_store = VectorStoreService()
        stats = vector_store.get_stats()
        
        stats_message = f"""📊 **Knowledge Base Statistics**

- **Collection**: {stats['collection_name']}
- **Documents indexed**: {stats['document_count']}
- **Embedding dimension**: {stats['embedding_dimension']}
"""
        await cl.Message(content=stats_message).send()
        
    except Exception as e:
        await cl.Message(content=f"Error getting stats: {str(e)}").send()
