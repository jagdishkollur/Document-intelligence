import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.retriever import answer_question, load_vector_collection


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --- Load once at startup, not per-request (same reasoning as Bike Demand's
# serve.py loading model.pkl once at import time, not inside the endpoint). ---
config = load_config()

app = FastAPI(title="Document Intelligence API")


# --- Pydantic models: define the accepted shape of requests/responses,
# so FastAPI validates BEFORE our function bodies ever run (R5 Q2/Q4). ---
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class DocumentsResponse(BaseModel):
    documents: list[str]


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    """
    GET — no side effects, just reports the API is alive.
    Same purpose as a health check in any deployed service: lets n8n or
    a load balancer confirm the process is up before sending real traffic.
    """
    return HealthResponse(status="ok")


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    """
    GET — lists the distinct source filenames currently stored in the
    Chroma collection (NOT the data/raw/ folder — see R5 Q8: the
    collection reflects what was actually ingested, which can drift
    from what's currently sitting on disk).
    """
    try:
        collection, _ = load_vector_collection(config)
        all_items = collection.get()  # returns all stored records + metadata
        metadatas = all_items.get("metadatas", [])

        sources = set()
        for meta in metadatas:
            source = meta.get("source")
            if source:
                sources.add(source)

        return DocumentsResponse(documents=sorted(sources))

    except Exception as e:
        # Collection may not exist yet if ingestion has never been run
        # (same NotFoundError we hit in R4 debugging).
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not ready: {e}",
        )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    POST — accepts a question, runs the full R4 RAG pipeline via
    answer_question(), and returns the grounded answer.

    request.question is guaranteed to exist and be a string by the time
    this function body runs — Pydantic already validated it (R5 Q4).
    We still guard against an empty/whitespace-only string, since that
    passes type validation but is semantically useless (R5 Q9).
    """
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
        # Catches: missing/empty Chroma collection, Groq API failures,
        # or any other runtime error — so one bad request can't take
        # down the whole running server process (R5 Q9).
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {e}",
        )