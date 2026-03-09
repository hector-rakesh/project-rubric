import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict]:
    """Discover available ChromaDB backends and format for Streamlit selection"""
    backends = {}
    current_dir = Path(".")
    
    # Look for directories containing Chroma data
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and (d / "chroma.sqlite3").exists()]

    for db_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(path=str(db_dir))
            collections = client.list_collections()
            
            for col in collections:
                # Key used for selection logic
                key = f"{db_dir.name}/{col.name}"
                count = col.count()
                
                backends[key] = {
                    "directory": str(db_dir),
                    "collection_name": col.name,
                    "display_name": f"{db_dir.name} > {col.name} ({count} docs)",
                    "count": count
                }
        except Exception as e:
            continue
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the ChromaDB collection"""
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        return collection, True, ""
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 5, 
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve documents with metadata filtering"""
    where_filter = None
    if mission_filter and mission_filter != "All Missions":
        where_filter = {"mission": mission_filter}

    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Improved context: Removes duplicates and orders by relevance"""
    if not documents:
        return "No relevant context found."
    
    seen_content = set()
    context_parts = ["--- START OF NASA MISSION CONTEXT ---"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        # Deduplication based on a snippet of the text
        content_hash = doc[:100].strip()
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        mission = meta.get("mission", "NASA General").replace("_", " ").title()
        source = meta.get("source", "Official Record")
        
        header = f"\n[Document {len(seen_content)} | Mission: {mission} | Source: {source}]"
        context_parts.append(header)
        context_parts.append(doc.strip())

    context_parts.append("\n--- END OF CONTEXT ---")
    return "\n".join(context_parts)