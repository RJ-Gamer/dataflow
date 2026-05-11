import time

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.services.ingest import get_vector_store

settings = get_settings()


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )


PROMPT_TEMPLATE = """
You are a helpful assistant for {project_name}.
Answer ONLY based on the provided context below.
If the answer is not in the context, say "I don't know based on the provided documentation."
Do not make up answers. Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(
    project_id: str,
    project_name: str,
    question: str,
    retrieval_k: int = 4,
    temperature: float = 0.0,
) -> dict:

    # --- Step 1: Set up retriever ---
    vector_store = get_vector_store(project_id)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": retrieval_k}
    )

    # --- Step 2: Build the RAG chain ---
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["project_name", "context", "question"],
    )

    llm = get_llm()

    # LCEL chain - read this carefully
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "project_name": lambda _: project_name,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- Step 3: Track with MLflow ---
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"project_{project_id}_questions")

    with mlflow.start_run(run_name=f"ask_{project_id[:8]}") as run:

        # Retrieve docs separately to measure latency + capture sources
        retrieval_start = time.time()
        source_docs = retriever.invoke(question)
        retrieval_latency = int((time.time() - retrieval_start) * 1000)

        # Run LLM
        llm_start = time.time()
        answer = chain.invoke(question)
        llm_latency = int((time.time() - llm_start) * 1000)

        # Log to MLflow
        mlflow.log_params(
            {
                "retrieval_k": retrieval_k,
                "temperature": temperature,
                "embedding_model": "gemini-embedding-001",
                "llm_model": "gemini-2.5-flash",
            }
        )

        mlflow.log_metrics(
            {
                "retrieval_latency_ms": retrieval_latency,
                "llm_latency_ms": llm_latency,
                "source_docs_count": len(source_docs),
            }
        )

        mlflow.log_text(question, "question.txt")
        mlflow.log_text(answer, "answer.txt")

        run_id = run.info.run_id

    # --- Step 4: Format sources ---
    sources = [
        {
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
        }
        for doc in source_docs
    ]

    return {
        "answer": answer,
        "sources": sources,
        "mlflow_run_id": run_id,
        "retrieval_latency_ms": retrieval_latency,
        "llm_latency_ms": llm_latency,
    }
