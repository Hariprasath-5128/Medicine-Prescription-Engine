"""Tests for the parts that must not silently regress.

    pytest tests/ -v

Tests needing the trained recommender skip cleanly when it is absent, so this
suite is useful from the first commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def retriever(cfg):
    from src.rag.retriever import EvidenceRetriever

    if not (ROOT / cfg["paths"]["chroma"]).exists():
        pytest.skip("evidence index not set up")
    return EvidenceRetriever(cfg)


class TestDatasetIntegrity:
    """The 50k dataset failed exactly these checks. Guard the replacements."""

    def test_processed_splits_do_not_share_text(self, cfg):
        import pandas as pd

        proc = ROOT / cfg["paths"]["processed"]
        if not (proc / "recommender_train.csv").exists():
            pytest.skip("run scripts/02_prepare_data.py --recommender")

        train = pd.read_csv(proc / "recommender_train.csv")
        test = pd.read_csv(proc / "recommender_test.csv")
        overlap = set(train["review"]) & set(test["review"])
        assert not overlap, f"{len(overlap)} reviews appear in both train and test"

    def test_training_data_has_real_variation(self, cfg):
        """Guards against a templated dataset sneaking back in."""
        import pandas as pd

        proc = ROOT / cfg["paths"]["processed"]
        if not (proc / "recommender_train.csv").exists():
            pytest.skip("run scripts/02_prepare_data.py --recommender")

        train = pd.read_csv(proc / "recommender_train.csv")
        distinct_ratio = train["review"].nunique() / len(train)
        assert distinct_ratio > 0.9, (
            f"only {distinct_ratio:.1%} of reviews are distinct - this looks "
            f"templated, like the 50k dataset that was rejected"
        )
        assert train["condition"].nunique() >= 20


class TestRetriever:
    def test_returns_evidence_with_provenance(self, retriever):
        result = retriever.retrieve("what causes keratoderma with woolly hair")
        assert result.is_grounded
        top = result.evidence[0]
        assert top.text
        assert top.source_url.startswith("http")
        assert 0.0 <= top.relevance <= 1.0

    def test_citation_is_numbered_and_linked(self, retriever):
        result = retriever.retrieve("asthma treatment")
        if not result.is_grounded:
            pytest.skip("no evidence for probe query")
        citation = result.evidence[0].citation(1)
        assert citation.startswith("[1]")

    def test_distance_threshold_filters(self, retriever):
        """A near-zero threshold must reject everything, not fail open."""
        result = retriever.retrieve("asthma treatment", max_distance=0.001)
        assert not result.evidence
        assert result.filtered_all

    def test_nonsense_query_is_not_force_matched(self, retriever):
        """Off-topic text should not be handed a confident citation."""
        result = retriever.retrieve(
            "quarterly consolidated revenue guidance for semiconductor fabrication"
        )
        if result.evidence:
            assert result.evidence[0].relevance < 0.75


class TestSafety:
    def test_red_flag_terms_detected(self, cfg):
        from src.rag.pipeline import PrescriptionPipeline

        pipe = PrescriptionPipeline(cfg, load_recommender=False)
        flags = pipe.check_red_flags("I have chest pain and shortness of breath")
        assert "chest pain" in flags
        assert "shortness of breath" in flags

    def test_benign_text_has_no_red_flags(self, cfg):
        from src.rag.pipeline import PrescriptionPipeline

        pipe = PrescriptionPipeline(cfg, load_recommender=False)
        assert pipe.check_red_flags("mild itchy skin rash on my arm") == []

    def test_disclaimer_is_configured(self, cfg):
        text = cfg["safety"]["disclaimer"].lower()
        assert "not a prescription" in text
        assert "clinician" in text


class TestRecommendation:
    def test_ungrounded_recommendation_says_so(self, cfg):
        """An empty evidence list must render as UNVERIFIED, never as silence."""
        from src.rag.pipeline import Recommendation

        rec = Recommendation(
            query="test", condition="Asthma", condition_confidence=0.9,
            drugs=[("Salbutamol", 0.8)], evidence=[], grounded=False,
            disclaimer=cfg["safety"]["disclaimer"],
        )
        rendered = rec.render()
        assert "UNVERIFIED" in rendered
        assert "NONE FOUND" in rendered

    def test_red_flag_dominates_the_output(self, cfg):
        from src.rag.pipeline import Recommendation

        rec = Recommendation(
            query="chest pain", condition="Heart Disease", condition_confidence=0.7,
            drugs=[("Aspirin", 0.6)], red_flag=True, red_flag_terms=["chest pain"],
            disclaimer=cfg["safety"]["disclaimer"],
        )
        rendered = rec.render()
        assert "URGENT" in rendered
        # The warning must come before the medication list.
        assert rendered.index("URGENT") < rendered.index("Aspirin")


class TestRecommenderModel:
    def test_predicts_when_trained(self, cfg):
        from src.models.recommender import RecommenderPipeline

        model_dir = ROOT / cfg["recommender"]["output_dir"]
        if not (model_dir / "pytorch_model.bin").exists():
            pytest.skip("recommender not trained")

        out = RecommenderPipeline(model_dir).predict(
            "I have been feeling very anxious and cannot sleep at night"
        )
        assert out.condition
        assert 0.0 <= out.condition_confidence <= 1.0
        assert len(out.drugs) > 0
        scores = [s for _, s in out.drugs]
        assert scores == sorted(scores, reverse=True), "drugs must be rank-ordered"
