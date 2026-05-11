from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ask, documents, projects
from app.core.config import get_settings
from app.db import models
from app.db.database import engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.Base.metadata.create_all(bind=engine)
    print(f"Starting {settings.app_name} with debug={settings.debug}")
    yield
    print(f"Shutting down {settings.app_name}")


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(ask.router)


@app.get("/")
def root():
    return {"message": f"Welcome to {settings.app_name}!"}
