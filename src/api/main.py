"""FastAPI backend for the Medicine Prescription Engine.

    uvicorn src.api.main:app --reload --port 8000

Endpoints
    GET  /health              readiness of each component
    POST /recommend           full pipeline: symptoms -> drugs + citations
    POST /evidence            retrieval only, for inspecting the corpus
    GET  /evidence/{chunk_id} one chunk with its full provenance
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
