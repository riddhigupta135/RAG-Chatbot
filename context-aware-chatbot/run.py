#!/usr/bin/env python
"""
Convenience script to run the RAG chatbot.
Provides commands to start the API, chat UI, or ingest documents.
"""

import argparse
import asyncio
import subprocess
import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Start the FastAPI backend server."""
    print(f"[*] Starting API server at http://{host}:{port}")
    print(f"[*] API docs available at http://{host}:{port}/docs")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
    ]
    
    if reload:
        cmd.append("--reload")
        
    subprocess.run(cmd)


def run_chat():
    """Start the Chainlit chat interface."""
    print("[*] Starting Chainlit chat interface...")
    subprocess.run([sys.executable, "-m", "chainlit", "run", "chainlit_app.py"])


def ingest_docs(source: str, source_type: str = "directory", follow_links: bool = False):
    """Ingest documents into the vector store."""
    print(f"[*] Ingesting documents from: {source}")
    
    # Import here to avoid slow startup for other commands
    from app.services.ingestion import IngestionService
    from app.models.schemas import IngestType
    from app.utils.logging import setup_logging
    
    setup_logging()
    
    service = IngestionService()
    
    # Map string type to enum
    type_map = {
        "directory": IngestType.DIRECTORY,
        "file": IngestType.FILE,
        "url": IngestType.URL,
        "text": IngestType.TEXT,
    }
    
    ingest_type = type_map.get(source_type.lower())
    if not ingest_type:
        print(f"[ERROR] Unknown source type: {source_type}")
        print(f"        Valid types: {', '.join(type_map.keys())}")
        sys.exit(1)
    
    # Run ingestion
    result = asyncio.run(service.ingest(
        ingest_type=ingest_type,
        source=source,
        follow_links=follow_links,
    ))
    
    if result["success"]:
        print(f"[OK] Ingestion complete!")
        print(f"     Documents processed: {result['documents_processed']}")
        print(f"     Chunks created: {result['chunks_created']}")
    else:
        print(f"[ERROR] Ingestion failed: {result['message']}")
        sys.exit(1)


def check_ollama():
    """Check if Ollama is running and has the required model."""
    print("[*] Checking Ollama status...")
    
    try:
        import httpx
        from app.config import get_settings
        
        settings = get_settings()
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"].split(":")[0] for m in models]
            
            print(f"[OK] Ollama is running")
            print(f"     Available models: {', '.join(model_names) or 'None'}")
            
            if settings.ollama_model not in model_names:
                print(f"[WARN] Required model '{settings.ollama_model}' not found")
                print(f"       Run: ollama pull {settings.ollama_model}")
            else:
                print(f"[OK] Required model '{settings.ollama_model}' is available")
        else:
            print(f"[ERROR] Ollama returned status {response.status_code}")
            
    except Exception as e:
        print(f"[ERROR] Cannot connect to Ollama: {e}")
        print("        Make sure Ollama is installed and running")
        print("        Install from: https://ollama.ai")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py api                    # Start the API server
  python run.py chat                   # Start the chat interface
  python run.py ingest ./data          # Ingest documents from directory
  python run.py ingest --type url https://docs.example.com
  python run.py check                  # Check Ollama status
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # Chat command
    subparsers.add_parser("chat", help="Start the Chainlit chat interface")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("source", help="Source path or URL")
    ingest_parser.add_argument(
        "--type", "-t",
        default="directory",
        choices=["directory", "file", "url", "text"],
        help="Type of source",
    )
    ingest_parser.add_argument(
        "--follow-links", "-f",
        action="store_true",
        help="Follow links when ingesting URLs",
    )
    
    # Check command
    subparsers.add_parser("check", help="Check Ollama status")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api(args.host, args.port, not args.no_reload)
    elif args.command == "chat":
        run_chat()
    elif args.command == "ingest":
        ingest_docs(args.source, args.type, args.follow_links)
    elif args.command == "check":
        check_ollama()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
