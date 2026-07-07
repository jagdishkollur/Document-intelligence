import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.retriever import answer_question, load_vector_collection
from src.ingest import run_ingestion
from src.tracker import timed_ingestion


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config()
app = FastAPI(title="Document Intelligence API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class DocumentsResponse(BaseModel):
    documents: list[str]


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    status: str
    num_chunks: int
    num_documents: int


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    try:
        collection, _ = load_vector_collection(config)
        all_items = collection.get()
        metadatas = all_items.get("metadatas", [])

        sources = set()
        for meta in metadatas:
            source = meta.get("source")
            if source:
                sources.add(source)

        return DocumentsResponse(documents=sorted(sources))

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not ready: {e}",
        )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="question must not be empty.",
        )

    try:
        answer = answer_question(question, config)
        return QueryResponse(answer=answer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {e}",
        )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Triggers the same incremental ingestion pipeline run_pipeline.py
    runs manually -- only NEW or CHANGED files (by content hash) get
    (re)processed, and the run is logged to MLflow automatically.

    This is the endpoint n8n's workflow will call whenever a new file
    lands in the watched folder (R7).
    """
    try:
        num_chunks, num_documents = timed_ingestion(run_ingestion, config)
        return IngestResponse(
            status="success",
            num_chunks=num_chunks,
            num_documents=num_documents,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}",
        )