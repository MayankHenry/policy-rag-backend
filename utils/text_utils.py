import fitz  # PyMuPDF
import docx
from typing import List

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    """Extract all text from a DOCX file."""
    doc = docx.Document(file_path)
    full_text = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(full_text).strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of tokens/words.

    Args:
        text: The raw document text.
        chunk_size: Number of words per chunk.
        overlap: Number of words to overlap between chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks
