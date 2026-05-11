from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.dependancies import get_project_by_api_key
from app.db.database import get_db
from app.db.models import Project, Question
from app.services.rag import ask_question

router = APIRouter(prefix="/projects", tags=["Ask"])


class AskRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    mlflow_run_id: str
    retrieval_latency_ms: int
    llm_latency_ms: int


@router.post("/{project_id}/ask", response_model=AskResponse)
def ask(
    project_id: str,
    body: AskRequest,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_by_api_key),
):
    if project.id != project_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask_question(
            project_id=project_id,
            project_name=project.name,
            question=body.question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")

    # Save to DB
    q = Question(
        project_id=project_id,
        question_text=body.question,
        answer_text=result["answer"],
        mlflow_run_id=result["mlflow_run_id"],
        retrieval_latency_ms=result["retrieval_latency_ms"],
        llm_latency_ms=result["llm_latency_ms"],
    )
    db.add(q)
    db.commit()

    return AskResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        mlflow_run_id=result["mlflow_run_id"],
        retrieval_latency_ms=result["retrieval_latency_ms"],
        llm_latency_ms=result["llm_latency_ms"],
    )
