"""Evidence retrieval with full provenance, over the SERA chunk index.

Every retrieved chunk keeps the chain you specified:

    recommendation -> evidence chunk -> original source URL

A note on honesty, which matters more here than in a normal RAG system: the
corpus is disease-information text (MedQuAD / Genetics Home Reference), not a
drug-trial corpus. Many drug-specific queries have no supporting chunk. This
module returns an empty list in that case rather than the least-bad match,
because a citation that does not support the claim is worse than no citation.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Evidence:
    """One retrieved chunk plus everything needed to cite it."""

    chunk_id: str
    text: str
    distance: float
    document_id: str = ""
    section: str = ""
    question_focus: str = ""
    umls_semantic_group: str = ""
    source_url: str = ""

    @property
    def relevance(self) -> float:
        """Cosine distance -> a 0-1 score that reads naturally in the UI."""
        return round(max(0.0, 1.0 - self.distance), 3)

    def citation(self, index: int) -> str:
        """Render as a numbered citation, e.g. `[1] Genetics Home Reference ...`."""
        title = self.question_focus.title() if self.question_focus else self.document_id
        parts = [f"[{index}] {title}"]
        if self.section:
            parts.append(f"Section: {self.section.title()}")
        if self.source_url:
            parts.append(self.source_url)
        return " | ".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["relevance"] = self.relevance
        return d


@dataclass
class RetrievalResult:
    query: str
    evidence: list[Evidence] = field(default_factory=list)
    # True when the index returned neighbours but all were above max_distance.
    filtered_all: bool = False

    @property
    def is_grounded(self) -> bool:
        return bool(self.evidence)

    def format_citations(self) -> str:
        if not self.evidence:
            return "No supporting evidence found in the indexed corpus."
        return "\n".join(e.citation(i) for i, e in enumerate(self.evidence, 1))


class EvidenceRetriever:
    """Thin, lazily-initialised wrapper over the copied SERA Chroma index."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or _load_config()
        ev = self.cfg["evidence"]
        self.collection_name = ev["collection"]
        self.top_k = ev["top_k"]
        self.max_distance = ev["max_distance"]
        self.model_name = ev["embedding_model"]
        self.device = ev.get("embedding_device", "cpu")
        self.chroma_dir = ROOT / self.cfg["paths"]["chroma"]
        self._collection = None
        self._embedder = None

    # Loading bge-m3 costs ~2.2 GB, so defer it until a query actually arrives.
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            print(f"loading embedder {self.model_name} on {self.device} ...")
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            if not self.chroma_dir.exists():
                raise FileNotFoundError(
                    f"evidence index missing at {self.chroma_dir}. "
                    "Run: python scripts/00_setup_evidence_db.py"
                )
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = client.get_collection(self.collection_name)
        return self._collection

    def retrieve(self, query: str, top_k: int | None = None,
                 max_distance: float | None = None) -> RetrievalResult:
        k = top_k or self.top_k
        limit = self.max_distance if max_distance is None else max_distance

        # The index was built with bge-m3, so the query must use the same model.
        vector = self.embedder.encode([query], normalize_embeddings=True).tolist()
        raw = self.collection.query(
            query_embeddings=vector,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        kept: list[Evidence] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            if dist > limit:
                continue
            meta = meta or {}
            kept.append(Evidence(
                chunk_id=cid,
                text=(doc or "").strip(),
                distance=round(float(dist), 4),
                document_id=str(meta.get("document_id", "")),
                section=str(meta.get("section", "")),
                question_focus=str(meta.get("question_focus", "")),
                umls_semantic_group=str(meta.get("umls_semantic_group", "")),
                source_url=str(meta.get("source_url", "")),
            ))

        return RetrievalResult(
            query=query,
            evidence=kept,
            filtered_all=bool(ids) and not kept,
        )

    def retrieve_for_recommendation(self, condition: str, drug: str) -> RetrievalResult:
        """Look for evidence about the condition being treated.

        Deliberately queries the *condition*, not "<drug> for <condition>": the
        corpus describes diseases, so a drug-shaped query mostly retrieves noise.
        """
        return self.retrieve(f"{condition} treatment and management")


@functools.lru_cache(maxsize=1)
def _load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "keratoderma with woolly hair causes"
    result = EvidenceRetriever().retrieve(q)
    print(f"\nquery: {q}")
    print(f"grounded: {result.is_grounded}  (all filtered: {result.filtered_all})\n")
    for i, ev in enumerate(result.evidence, 1):
        print(f"--- [{i}] relevance={ev.relevance} section={ev.section}")
        print(f"    {ev.text[:220]}...")
        print(f"    source: {ev.source_url}\n")
