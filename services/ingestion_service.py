from fastapi import UploadFile, HTTPException
from utils.file_utils import save_upload_file
from utils.text_utils import extract_text_from_pdf, extract_text_from_docx, chunk_text
from utils.faiss_utils import load_faiss_index, save_faiss_index
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load embedding model once at startup
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def ingest_document(file: UploadFile) -> dict:
    """Save uploaded file, extract text, chunk, embed, store in FAISS."""
    allowed_types = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx"
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
    
    # Save file locally
    file_path = save_upload_file(file)
    extension = allowed_types[file.content_type]

    # Extract text
    if extension == "pdf":
        text = extract_text_from_pdf(file_path)
    elif extension == "docx":
        text = extract_text_from_docx(file_path)

    # Chunk text
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    # Generate embeddings
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # Load or create FAISS index
    index, metadata = load_faiss_index(embeddings.shape[1])

    # Add to index
    index.add(embeddings)

    # Prepare metadata: (doc_name, chunk_text)
    for chunk in chunks:
        metadata.append({
            "filename": file.filename,
            "text": chunk
        })

    # Save updated FAISS index
    save_faiss_index(index, metadata)

    return {
        "filename": file.filename,
        "path": file_path,
        "total_chunks": len(chunks),
        "vector_dim": embeddings.shape[1],
        "message": f"Document processed and stored in FAISS with {len(chunks)} chunks."
    }
