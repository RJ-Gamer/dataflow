import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.dependancies import get_project_by_api_key
from app.db.database import get_db
from app.db.models import Document, Project, Question

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    api_key: str
    created_at: str

    class Config:
        from_attributes = True


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    api_key = secrets.token_hex(32)

    project = Project(
        name=project.name, description=project.description, api_key=api_key
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        api_key=project.api_key,
        created_at=project.created_at.isoformat(),
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        api_key=project.api_key,
        created_at=project.created_at.isoformat(),
    )


@router.get("/", response_model=list[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    return [
        ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            api_key=project.api_key,
            created_at=project.created_at.isoformat(),
        )
        for project in projects
    ]


@router.get("/{project_id}/analytics")
def get_analytics(
    project_id: str,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_by_api_key),
):

    if project.id != project_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Document stats
    docs = (
        db.query(Document)
        .filter(Document.project_id == project_id, Document.is_processed == True)
        .all()
    )

    total_chunks = sum(d.chunk_count for d in docs)

    # Question stats
    questions = db.query(Question).filter(Question.project_id == project_id).all()

    avg_retrieval_ms = (
        sum(q.retrieval_latency_ms for q in questions if q.retrieval_latency_ms)
        / len(questions)
        if questions
        else 0
    )

    avg_llm_ms = (
        sum(q.llm_latency_ms for q in questions if q.llm_latency_ms) / len(questions)
        if questions
        else 0
    )

    # Recent questions
    recent = (
        db.query(Question)
        .filter(Question.project_id == project_id)
        .order_by(Question.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "project": {
            "id": project.id,
            "name": project.name,
        },
        "documents": {
            "total": len(docs),
            "total_chunks": total_chunks,
            "files": [
                {
                    "filename": d.filename,
                    "chunk_count": d.chunk_count,
                    "created_at": str(d.created_at),
                }
                for d in docs
            ],
        },
        "questions": {
            "total_asked": len(questions),
            "avg_retrieval_latency_ms": round(avg_retrieval_ms),
            "avg_llm_latency_ms": round(avg_llm_ms),
        },
        "recent_questions": [
            {
                "question": q.question_text,
                "answer": q.answer_text,
                "mlflow_run_id": q.mlflow_run_id,
                "asked_at": str(q.created_at),
            }
            for q in recent
        ],
    }
