import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # 1. Create list of directories that match specific criteria
    # We look for directories that contain a 'chroma.sqlite3' file
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and (d / "chroma.sqlite3").exists()]

    # 2. Loop through each discovered directory
    for db_dir in chroma_dirs:
        try:
            # 3. Initialize database client
            client = chromadb.PersistentClient(path=str(db_dir))
            
            # 4. Retrieve list of available collections
            collections = client.list_collections()
            
            # 5. Loop through each collection found
            for col in collections:
                # Create unique identifier: "directory/collection"
                key = f"{db_dir.name}/{col.name}"
                
                # Get document count with fallback
                try:
                    count = col.count()
                except Exception:
                    count = "Unknown"

                # 6. Build information dictionary
                backends[key] = {
                    "path": str(db_dir),
                    "collection": col.name,
                    "display_name": f"{db_dir.name} > {col.name} ({count} docs)",
                    "count": count
                }
        
        except Exception as e:
            # 7. Handle connection errors gracefully
            error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
            backends[str(db_dir)] = {
                "path": str(db_dir),
                "collection": "N/A",
                "display_name": f"⚠️ {db_dir.name} (Error: {error_msg})",
                "count": 0
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend"""
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(name=collection_name)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    
    # 1. Initialize filter variable
    where_filter = None

    # 2. Check for filter parameters (ignoring "all" or empty values)
    if mission_filter and mission_filter.lower() not in ["all", "none", "any"]:
        where_filter = {"mission": mission_filter}

    # 3. Execute database query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter  # Apply conditional filter
    )

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # 1. Initialize context list
    context_parts = ["--- START OF SEARCH CONTEXT ---"]

    # 2. Loop through paired documents and metadata
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        # Extract and clean Mission info
        mission = meta.get("mission", "Unknown Mission").replace("_", " ").title()
        
        # Extract Source and Category
        source = meta.get("source", "Unknown Source")
        category = meta.get("category", "General").replace("_", " ").title()
        
        # 3. Create formatted source header
        header = f"\n[Source {i} | Mission: {mission} | Category: {category} | File: {source}]"
        context_parts.append(header)
        
        # 4. Truncate document if it's excessively long (e.g., 2000 chars)
        content = doc if len(doc) < 2000 else doc[:2000] + "... [Content Truncated]"
        context_parts.append(content)

    context_parts.append("\n--- END OF SEARCH CONTEXT ---")
    
    return "\n".join(context_parts)