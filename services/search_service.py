import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

vectorizer = None

def semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for similar chunks using TF-IDF and cosine similarity"""
    global vectorizer
    storage_dir = "data/vector_store"
    
    if not os.path.exists(storage_dir):
        return []

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

            # Create vectorizer if not exists
            if vectorizer is None:
                vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
                vectorizer.fit(chunks)  # Fit on document chunks

            # Get query embedding
            query_embedding = vectorizer.transform([query]).toarray()

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
