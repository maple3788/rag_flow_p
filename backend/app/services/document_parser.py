from io import BytesIO
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from fastapi import HTTPException, UploadFile

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


async def parse_uploaded_file(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if suffix == ".pdf":
        return _sanitize_text(_extract_pdf_text(data))
    if suffix == ".docx":
        return _sanitize_text(_extract_docx_text(data))
    return _sanitize_text(data.decode("utf-8", errors="ignore"))


def _extract_pdf_text(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
    text = "\n".join(pages).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    return text


def _extract_docx_text(data: bytes) -> str:
    with BytesIO(data) as buffer:
        doc = DocxDocument(buffer)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from DOCX")
    return text


def _sanitize_text(text: str) -> str:
    # PostgreSQL text fields reject NUL bytes.
    return text.replace("\x00", "")
