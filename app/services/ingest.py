import time
from pathlib import Path

import mlflow
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings

settings = get_settings()


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", google_api_key=settings.google_api_key
    )


def get_vector_store(project_id: str):
    return Chroma(
        collection_name=f"project_{project_id}",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def ingest_document(
    project_id: str,
    file_path: str,
    file_type: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:

    # Use pathlib for reliable cross-platform path handling
    resolved_path = Path(file_path).resolve()

    # Debug — print to server console so we can see the actual path
    print(f"[DEBUG] Resolved path: {resolved_path}")
    print(f"[DEBUG] File exists: {resolved_path.exists()}")

    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")

    # --- Step 1: Load document ---
    if file_type == "pdf":
        loader = PyPDFLoader(str(resolved_path))
    elif file_type == "txt":
        loader = TextLoader(str(resolved_path), autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    documents = loader.load()

    # --- Step 2: Chunk it ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # --- Step 3: Embed and store ---
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"project_{project_id}_ingestion")

    with mlflow.start_run(run_name=f"ingest_{file_type}"):
        start = time.time()

        vector_store = get_vector_store(project_id)
        vector_store.add_documents(chunks)

        elapsed_ms = int((time.time() - start) * 1000)

        mlflow.log_params(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "file_type": file_type,
                "embedding_model": "gemini-embedding-001",
            }
        )

        mlflow.log_metrics(
            {
                "total_chunks": len(chunks),
                "total_pages": len(documents),
                "embedding_latency_ms": elapsed_ms,
            }
        )

    return {
        "chunk_count": len(chunks),
        "page_count": len(documents),
        "embedding_latency_ms": elapsed_ms,
    }
