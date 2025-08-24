import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from docx import Document
import os
import json
from typing import List, Dict, Any

# Initialize the SentenceTransformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

def create_embeddings(texts: List[str]) -> np.ndarray:
    """Create embeddings using SentenceTransformer"""
    if len(texts) == 0:
        return np.array([])
    
    embeddings = embedding_model.encode(texts, 
                                     convert_to_numpy=True,
                                     normalize_embeddings=True)
    return embeddings

def ingest_document(file_path: str, filename: str) -> Dict[str, Any]:
    """Process document and create embeddings"""
    
    # Extract text based on file type
    if filename.lower().endswith('.pdf'):
        text_chunks = extract_pdf_text(file_path)
    elif filename.lower().endswith('.docx'):
        text_chunks = extract_docx_text(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    if text_chunks:
        embeddings = create_embeddings(text_chunks)
        save_to_local_storage(filename, text_chunks, embeddings)
        
        return {
            "filename": filename,
            "total_chunks": len(text_chunks),
            "status": "success"
        }
    else:
        raise ValueError("No text extracted from document")

def extract_pdf_text(file_path: str) -> List[str]:
    """Extract text from PDF"""
    doc = fitz.open(file_path)
    text_chunks = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        
        # Simple chunking by paragraphs
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        text_chunks.extend(chunks)
    
    doc.close()
    return text_chunks

def extract_docx_text(file_path: str) -> List[str]:
    """Extract text from DOCX"""
    doc = Document(file_path)
    text_chunks = []
    
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_chunks.append(paragraph.text.strip())
    
    return text_chunks

def save_to_local_storage(filename: str, chunks: List[str], embeddings: np.ndarray):
    """Save chunks and embeddings to JSON file"""
    storage_dir = "data/vector_store"
    os.makedirs(storage_dir, exist_ok=True)
    
    storage_file = os.path.join(storage_dir, f"{filename}_data.json")
    
    data = {
        "filename": filename,
        "chunks": chunks,
        "embeddings": embeddings.tolist()
    }
    
    with open(storage_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
