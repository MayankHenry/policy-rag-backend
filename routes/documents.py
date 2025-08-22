import os
from fastapi import APIRouter, HTTPException
from services.ingestion_service import ingest_document
from utils.faiss_utils import load_faiss_index, save_faiss_index

router = APIRouter()
UPLOAD_FOLDER = "data/uploaded_docs"

@router.post("/upload")
def upload_document(file):
    return ingest_document(file)

@router.get("/list")
def list_documents():
    """List all uploaded documents."""
    if not os.path.exists(UPLOAD_FOLDER):
        return []
    return os.listdir(UPLOAD_FOLDER)

@router.delete("/delete/{filename}")
def delete_document(filename: str):
    """Delete a document and remove its chunks from FAISS."""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from disk
    os.remove(file_path)

    # Remove chunks from FAISS
    index, metadata = load_faiss_index(dimension=384)  # assuming 384-dim embeddings

    new_metadata = []
    keep_indices = []

    for idx, meta in enumerate(metadata):
        if meta["filename"] != filename:
            new_metadata.append(meta)
            keep_indices.append(idx)

    # Rebuild FAISS index with only kept vectors
    import numpy as np
    vectors = []
    for idx in keep_indices:
        vectors.append(index.reconstruct(idx))
    if vectors:
        import faiss
        new_index = faiss.IndexFlatIP(len(vectors[0]))
        new_index.add(np.array(vectors))
    else:
        import faiss
        new_index = faiss.IndexFlatIP(384)

    save_faiss_index(new_index, new_metadata)

    return {"message": f"Deleted {filename} and its vectors from FAISS"}
