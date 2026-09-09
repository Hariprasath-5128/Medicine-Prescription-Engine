"""End-to-end pipeline: symptoms -> condition + drugs -> evidence -> citations.

    user text
       -> safety triage (red flags short-circuit everything)
       -> recommender  (condition + ranked drugs)
       -> retriever    (evidence chunks with source URLs)
       -> Recommendation with numbered citations

Design note on grounding: the recommender and the retriever are independent.
The retriever confirms or fails to confirm what the recommender proposed; it
never supplies the recommendation. When no chunk clears the distance threshold
the result is marked ungrounded and says so, rather than quietly presenting an
unsupported suggestion as evidence-backed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Support both `import src.rag.pipeline` and `python src/rag/pipeline.py`; the
# latter has no parent package, so a relative import would fail.
if __package__:
    from .retriever import Evidence, EvidenceRetriever
else:  # pragma: no cover - direct-execution path
    sys.path.insert(0, str(ROOT))
    from src.rag.retriever import Evidence, EvidenceRetriever


@dataclass
class Recommendation:
    query: str
    condition: str
    condition_confidence: float
    drugs: list[tuple[str, float]]
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    grounded: bool = False
    red_flag: bool = False
    red_flag_terms: list[str] = field(default_factory=list)
    disclaimer: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "condition": self.condition,
            "condition_confidence": self.condition_confidence,
            "drugs": [{"name": n, "score": s} for n, s in self.drugs],
            "alternatives": [{"name": n, "score": s} for n, s in self.alternatives],
            "evidence": [e.to_dict() for e in self.evidence],
            "grounded": self.grounded,
            "red_flag": self.red_flag,
            "red_flag_terms": self.red_flag_terms,
            "disclaimer": self.disclaimer,
        }

    def render(self) -> str:
        lines: list[str] = []
        if self.red_flag:
            lines += [
                "*** URGENT - SEEK IMMEDIATE MEDICAL ATTENTION ***",
                f"Your description mentions: {', '.join(self.red_flag_terms)}.",
                "These can indicate a medical emergency. Contact emergency services "
                "or go to an emergency department now.",
                "",
            ]
        lines.append(f"Likely condition : {self.condition} "
                     f"(confidence {self.condition_confidence:.1%})")
        if self.alternatives:
            alts = ", ".join(f"{n} ({s:.1%})" for n, s in self.alternatives)
            lines.append(f"Also considered  : {alts}")

        lines.append("")
        lines.append("Commonly associated medications (from patient-reported data):")
        for i, (name, score) in enumerate(self.drugs, 1):
            lines.append(f"  {i}. {name}  (association {score:.1%})")

        lines.append("")
        if self.grounded:
            lines.append("Supporting evidence:")
            for i, ev in enumerate(self.evidence, 1):
                lines.append(f"\n{ev.citation(i)}")
                lines.append(f'    "{ev.text[:300].strip()}..."')
                lines.append(f"    relevance: {ev.relevance}")
        else:
            lines.append("Supporting evidence: NONE FOUND.")
            lines.append("  No chunk in the indexed corpus passed the relevance "
                         "threshold, so this suggestion is UNVERIFIED.")

        lines += ["", "-" * 70, self.disclaimer]
        return "\n".join(lines)


class PrescriptionPipeline:
    def __init__(self, config: dict | None = None, load_recommender: bool = True):
        self.cfg = config or _load_config()
        self.safety = self.cfg["safety"]
        self.retriever = EvidenceRetriever(self.cfg)
        self._recommender = None
        if load_recommender:
            self._load_recommender()

    def _load_recommender(self):
        from src.models.recommender import RecommenderPipeline

        model_dir = ROOT / self.cfg["recommender"]["output_dir"]
        self._recommender = RecommenderPipeline(model_dir)
        return self._recommender

    @property
    def recommender(self):
        if self._recommender is None:
            self._load_recommender()
        return self._recommender

    def check_red_flags(self, text: str) -> list[str]:
        low = text.lower()
        return [t for t in self.safety["red_flag_terms"] if t in low]

    def run(self, query: str, top_k_drugs: int = 5) -> Recommendation:
        flags = self.check_red_flags(query)
        pred = self.recommender.predict(query, top_k_drugs=top_k_drugs)
        retrieved = self.retriever.retrieve_for_recommendation(
            pred.condition, pred.drugs[0][0] if pred.drugs else ""
        )
        return Recommendation(
            query=query,
            condition=pred.condition,
            condition_confidence=pred.condition_confidence,
            drugs=pred.drugs,
            alternatives=pred.alternatives,
            evidence=retrieved.evidence,
            grounded=retrieved.is_grounded,
            red_flag=bool(flags),
            red_flag_terms=flags,
            disclaimer=self.safety["disclaimer"].strip(),
        )


def _load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "I have been feeling anxious and cannot sleep"
    print(PrescriptionPipeline().run(q).render())
