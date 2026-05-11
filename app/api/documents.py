import os
import shutil
import traceback

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.dependancies import get_project_by_api_key
from app.db.database import get_db
from app.db.models import Document, Project
from app.services.ingest import ingest_document

router = APIRouter(prefix="/projects", tags=["Documents"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_TYPES = {"pdf", "txt"}


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    is_processed: bool
    created_at: str


@router.post("/{project_id}/ingest", response_model=DocumentResponse)
def ingest(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_by_api_key),
):
    # Confirm project_id matches authenticated project
    if project.id != project_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Validate file type
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_TYPES}"
        )

    # Save file to disk
    file_path = os.path.join(UPLOAD_DIR, f"{project_id}_{file.filename}").replace(
        "\\", "/"
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create DB record
    doc = Document(
        project_id=project_id,
        filename=file.filename,
        file_type=extension,
        is_processed=False,
        chunk_count=0,
    )
    db.add(doc)
    db.commit()

    # Run ingestion
    try:
        result = ingest_document(
            project_id=project_id,
            file_path=file_path,
            file_type=extension,
        )
        doc.is_processed = True
        doc.chunk_count = result["chunk_count"]
        db.commit()
        db.refresh(doc)

    except Exception as e:
        traceback.print_exc()  # prints full traceback to server console
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        chunk_count=doc.chunk_count,
        is_processed=doc.is_processed,
        created_at=str(doc.created_at),
    )


@router.get("/{project_id}/documents", response_model=list[DocumentResponse])
def list_documents(
    project_id: str,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_by_api_key),
):
    if project.id != project_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    docs = db.query(Document).filter(Document.project_id == project_id).all()

    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            file_type=d.file_type,
            chunk_count=d.chunk_count,
            is_processed=d.is_processed,
            created_at=str(d.created_at),
        )
        for d in docs
    ]


class IngestConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50


@router.post("/{project_id}/ingest/experiment", response_model=DocumentResponse)
def ingest_experiment(
    project_id: str,
    file: UploadFile = File(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_by_api_key),
):
    if project.id != project_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_path = os.path.join(
        UPLOAD_DIR, f"{project_id}_exp_{chunk_size}_{file.filename}"
    ).replace("\\", "/")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc = Document(
        project_id=project_id,
        filename=f"[exp-{chunk_size}] {file.filename}",
        file_type=extension,
        is_processed=False,
        chunk_count=0,
    )
    db.add(doc)
    db.commit()

    try:
        result = ingest_document(
            project_id=project_id,
            file_path=file_path,
            file_type=extension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        doc.is_processed = True
        doc.chunk_count = result["chunk_count"]
        db.commit()
        db.refresh(doc)

    except Exception as e:
        import traceback

        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        chunk_count=doc.chunk_count,
        is_processed=doc.is_processed,
        created_at=str(doc.created_at),
    )
