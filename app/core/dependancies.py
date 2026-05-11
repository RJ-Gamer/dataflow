from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db


def get_project_by_api_key(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    from app.db.models import Project

    project = db.query(Project).filter(Project.api_key == x_api_key).first()
    if not project:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return project
