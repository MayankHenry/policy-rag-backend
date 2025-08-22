import os
from pathlib import Path
from fastapi import UploadFile

UPLOAD_FOLDER = "data/uploaded_docs"

def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to local storage and return file path."""
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, upload_file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return file_path
