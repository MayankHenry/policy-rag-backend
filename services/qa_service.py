import os
import requests
from services.search_service import semantic_search
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b:free"

def generate_answer_with_openrouter(question: str, top_k: int = 5):
    """
    Use OpenRouter API with the provided model to answer a question based on 
    top_k relevant chunks retrieved from FAISS semantic search.
    """
    chunks = semantic_search(question, top_k=top_k)

    if not chunks:
        return {"answer": "No relevant information found.", "context": []}

    # Concatenate chunk texts as context
    context_text = "\n\n".join([c["text"] for c in chunks])
    prompt = f"""
You are an assistant answering questions based on the provided context:
Context:
{context_text}

Question:
{question}

Answer concisely and cite the context.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You answer based only on the given context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        answer_text = data["choices"][0]["message"]["content"].strip()
        return {
            "answer": answer_text,
            "context": chunks
        }
    except Exception as e:
        return {"answer": f"Error contacting OpenRouter API: {str(e)}", "context": []}
