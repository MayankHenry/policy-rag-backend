# services/search_service.py

from sentence_transformers import SentenceTransformer
from utils.faiss_utils import load_faiss_index
import numpy as np

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_search(query: str, top_k: int = 5):
    """
    Perform semantic search by embedding the query and retrieving
    the closest chunks from the FAISS index.
    """
    # Embed query and normalize (since FAISS index uses normalized embeddings)
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Load index and metadata
    index, metadata = load_faiss_index(dimension=query_embedding.shape[1])

    if index.ntotal == 0:
        return []  # No indexed documents

    # Search FAISS index (returns distances and indices)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "filename": meta["filename"],
            "text": meta["text"],
            "score": float(dist)  # similarity score
        })

    return results
