"""FastAPI backend for the Medicine Prescription Engine.

    uvicorn src.api.main:app --reload --port 8000

Endpoints
    GET  /health              readiness of each component
    POST /recommend           full pipeline: symptoms -> drugs + citations
    POST /evidence            retrieval only, for inspecting the corpus
    GET  /evidence/{chunk_id} one chunk with its full provenance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rag.pipeline import PrescriptionPipeline  # noqa: E402
from src.rag.retriever import EvidenceRetriever  # noqa: E402

with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

app = FastAPI(
    title="Medicine Prescription Engine",
    description="Evidence-grounded medication decision support. Not a prescriber.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Populated lazily so the server starts even when the recommender is untrained.
_pipeline: PrescriptionPipeline | None = None
_retriever: EvidenceRetriever | None = None


def get_retriever() -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever(CFG)
    return _retriever


def get_pipeline() -> PrescriptionPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = PrescriptionPipeline(CFG)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"{exc} Train it with scripts/03_train_recommender.py",
            ) from exc
    return _pipeline


class RecommendRequest(BaseModel):
    text: str = Field(..., min_length=3, description="symptom description")
    top_k_drugs: int = Field(5, ge=1, le=20)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """assistant-ui posts the whole thread; only the last user turn is used."""

    messages: list[ChatMessage]


class EvidenceRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)
    max_distance: float | None = Field(None, ge=0.0, le=2.0)


@app.get("/health")
def health() -> dict:
    chroma = ROOT / CFG["paths"]["chroma"]
    model_dir = ROOT / CFG["recommender"]["output_dir"]
    status = {
        "status": "ok",
        "evidence_db": chroma.exists(),
        "recommender_trained": (model_dir / "pytorch_model.bin").exists(),
    }
    if chroma.exists():
        try:
            status["chunks"] = get_retriever().collection.count()
        except Exception as exc:  # index present but unreadable
            status["evidence_db"] = False
            status["error"] = str(exc)
    return status


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict:
    return get_pipeline().run(req.text, top_k_drugs=req.top_k_drugs).to_dict()


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Conversational wrapper over the pipeline, streamed for the chat UI.

    Streams newline-delimited JSON: `{"type": "text", ...}` chunks build the
    visible answer, then a single `{"type": "data", ...}` frame carries the
    structured recommendation so the UI can render evidence cards and
    citations rather than re-parsing prose.
    """
    user_turns = [m for m in req.messages if m.role == "user"]
    if not user_turns:
        raise HTTPException(status_code=400, detail="no user message in thread")
    text = user_turns[-1].content.strip()
    if len(text) < 3:
        raise HTTPException(status_code=400, detail="message too short")

    def stream():
        try:
            rec = get_pipeline().run(text)
        except HTTPException as exc:
            yield json.dumps({"type": "text", "text": str(exc.detail)}) + "\n"
            return
        except Exception as exc:  # surface the failure in the thread
            yield json.dumps({"type": "text", "text": f"Error: {exc}"}) + "\n"
            return

        # Word-by-word so the UI shows progressive output rather than a jump.
        for word in rec.render().split(" "):
            yield json.dumps({"type": "text", "text": word + " "}) + "\n"
        yield json.dumps({"type": "data", "payload": rec.to_dict()}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/evidence")
def evidence(req: EvidenceRequest) -> dict:
    result = get_retriever().retrieve(req.query, req.top_k, req.max_distance)
    return {
        "query": result.query,
        "grounded": result.is_grounded,
        "filtered_all": result.filtered_all,
        "evidence": [e.to_dict() for e in result.evidence],
    }


@app.get("/evidence/{chunk_id}")
def get_chunk(chunk_id: str) -> dict:
    col = get_retriever().collection
    got = col.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not got["ids"]:
        raise HTTPException(status_code=404, detail=f"no chunk {chunk_id}")
    return {
        "chunk_id": chunk_id,
        "text": got["documents"][0],
        "metadata": got["metadatas"][0],
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "Medicine Prescription Engine",
        "docs": "/docs",
        "disclaimer": CFG["safety"]["disclaimer"].strip(),
    }
