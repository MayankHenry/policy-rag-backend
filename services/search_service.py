import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# Initialize the SentenceTransformer model with fixed max length
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

def semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for similar chunks using SentenceTransformer embeddings and cosine similarity"""
    storage_dir = "data/vector_store"
    
    if not os.path.exists(storage_dir):
        return []

    # Generate query embedding
    query_embedding = embedding_model.encode([query], 
                                          convert_to_numpy=True,
                                          normalize_embeddings=True)
    all_results = []

    # Load all stored document data
    for filename in os.listdir(storage_dir):
        if filename.endswith('_data.json'):
            file_path = os.path.join(storage_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            chunks = data['chunks']
            stored_embeddings = np.array(data['embeddings'])

            if len(chunks) == 0:
                continue

            # Normalize stored embeddings if needed
            stored_embeddings = stored_embeddings / np.linalg.norm(stored_embeddings, axis=1)[:, np.newaxis]

            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

            # Add results with similarity scores
            for i, similarity in enumerate(similarities):
                all_results.append({
                    'text': chunks[i],
                    'filename': data['filename'],
                    'similarity': float(similarity)
                })

    # Sort by similarity (highest first) and return top_k
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    return all_results[:top_k]