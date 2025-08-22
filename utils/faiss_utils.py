import faiss
import numpy as np
import os
from typing import List, Tuple

FAISS_INDEX_FILE = "data/vector_store/faiss_index.idx"
FAISS_METADATA_FILE = "data/vector_store/metadata.npy"

def save_faiss_index(index, metadata):
    """Save FAISS index and metadata to disk."""
    os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(FAISS_METADATA_FILE, np.array(metadata, dtype=object))

def load_faiss_index(dimension: int) -> Tuple[faiss.IndexFlatIP, list]:
    """Load FAISS index and metadata if exists, else create empty."""
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        metadata = np.load(FAISS_METADATA_FILE, allow_pickle=True).tolist()
    else:
        index = faiss.IndexFlatIP(dimension)  # cosine similarity
        metadata = []
    return index, metadata
