import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Policy RAG API", 
    version="1.0.0",
    description="Semantic search and QA over policy documents"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directories if they don't exist
os.makedirs("data/uploaded_docs", exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

# Serve static files (your HTML frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Import routes
from routes import documents, search, qa

app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(qa.router, prefix="/qa", tags=["QA"])

@app.get("/")
def root():
    return {"message": "Policy RAG API is running!", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}
