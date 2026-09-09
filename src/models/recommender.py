"""Condition -> drug recommender, trained on real patient review text.

Two heads over one PubMedBERT encoder:

  condition head  multi-class over 90 conditions (softmax)
  drug head       multi-label over the drugs actually prescribed for those
                  conditions (sigmoid) - a condition legitimately has several
                  valid drugs, so this is ranking, not classification

The multi-label drug head is what makes Precision@K / Recall@K meaningful.
A single-label head would force one "correct" drug per condition, which is
clinically wrong and inflates the metric.

Trains in ~3 GB, so it runs on the 4 GB dev card as well as the 12 GB box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RecommenderOutput:
    condition: str
    condition_confidence: float
    drugs: list[tuple[str, float]]          # (drug name, score), descending
    alternatives: list[tuple[str, float]]   # runner-up conditions


class DrugRecommender(nn.Module):
    """PubMedBERT encoder with a condition head and a drug-ranking head."""

    def __init__(self, encoder_name: str, n_conditions: int, n_drugs: int,
                 dropout: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.condition_head = nn.Linear(hidden, n_conditions)
        self.drug_head = nn.Linear(hidden, n_drugs)

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """Per-condition loss weights, registered so .to(device) moves them."""
        self.register_buffer("class_weights", weights)

    def forward(self, input_ids, attention_mask, condition_labels=None,
                drug_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over real tokens; more stable than [CLS] for a head trained
        # from scratch on a modest corpus.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)

        condition_logits = self.condition_head(pooled)
        drug_logits = self.drug_head(pooled)

        loss = None
        if condition_labels is not None:
            # Class weights + label smoothing both target macro-F1: the rare
            # conditions carry most of that metric's weight, and smoothing stops
            # the head becoming over-confident on the few dominant classes.
            loss = nn.functional.cross_entropy(
                condition_logits, condition_labels,
                weight=getattr(self, "class_weights", None),
                label_smoothing=self.label_smoothing,
            )
            if drug_labels is not None:
                # Equal weighting: both objectives share one encoder, and the
                # drug head is the harder task, so it must not be drowned out.
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    drug_logits, drug_labels.float()
                )
        return {"loss": loss, "condition_logits": condition_logits,
                "drug_logits": drug_logits}


class RecommenderPipeline:
    """Inference wrapper that loads a trained checkpoint and its label maps."""

    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"no trained recommender at {self.model_dir}. "
                "Run: python scripts/03_train_recommender.py"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        meta = json.loads((self.model_dir / "label_maps.json").read_text(encoding="utf-8"))
        self.conditions: list[str] = meta["conditions"]
        self.drugs: list[str] = meta["drugs"]
        self.max_length: int = meta.get("max_length", 256)
        encoder_name: str = meta["encoder"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = DrugRecommender(encoder_name, len(self.conditions), len(self.drugs))
        state = torch.load(self.model_dir / "pytorch_model.bin", map_location="cpu")
        # `class_weights` is a training-only buffer (see set_class_weights); a
        # model built for inference never registers it, so drop it rather than
        # failing the load on an unexpected key.
        state.pop("class_weights", None)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, text: str, top_k_drugs: int = 5,
                top_k_conditions: int = 3) -> RecommenderOutput:
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt").to(self.device)
        out = self.model(enc["input_ids"], enc["attention_mask"])

        cond_probs = torch.softmax(out["condition_logits"][0], dim=-1)
        drug_probs = torch.sigmoid(out["drug_logits"][0])

        c_scores, c_idx = cond_probs.topk(min(top_k_conditions, len(self.conditions)))
        d_scores, d_idx = drug_probs.topk(min(top_k_drugs, len(self.drugs)))

        return RecommenderOutput(
            condition=self.conditions[c_idx[0]],
            condition_confidence=round(c_scores[0].item(), 4),
            drugs=[(self.drugs[i], round(s.item(), 4))
                   for i, s in zip(d_idx.tolist(), d_scores)],
            alternatives=[(self.conditions[i], round(s.item(), 4))
                          for i, s in zip(c_idx.tolist()[1:], c_scores[1:])],
        )
