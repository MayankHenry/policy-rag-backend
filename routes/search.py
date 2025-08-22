from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from services.search_service import semantic_search

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5  # number of results to return

class SearchResult(BaseModel):
    filename: str
    text: str
    score: float  # similarity score

@router.post("/", response_model=List[SearchResult])
def search(request: SearchRequest):
    """
    Perform semantic search on uploaded documents.
    Returns top_k most relevant chunks matching the query.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    results = semantic_search(request.query, request.top_k)
    return results
