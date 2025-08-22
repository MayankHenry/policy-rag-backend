from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from services.qa_service import generate_answer_with_openrouter

router = APIRouter()

class QARequest(BaseModel):
    question: str
    top_k: int = 5

@router.post("/")
def qa_endpoint(request: QARequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    result = generate_answer_with_openrouter(request.question, top_k=request.top_k)
    return result
