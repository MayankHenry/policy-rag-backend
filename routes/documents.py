import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from services.ingestion_service import ingest_document

router = APIRouter()
UPLOAD_FOLDER = "data/uploaded_docs"

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        result = ingest_document(file_path, file.filename)
        
        return {
            "filename": file.filename,
            "total_chunks": result.get("total_chunks", 0),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    return {"message": f"Deleted {filename} successfully"}
