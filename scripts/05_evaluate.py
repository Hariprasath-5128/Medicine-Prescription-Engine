"""Evaluate the system end to end and write a report to artifacts/.

    python scripts/05_evaluate.py --retrieval          # RAG only, no training needed
    python scripts/05_evaluate.py --all

Metrics
    recommender : condition accuracy, macro-F1, drug Precision@K / Recall@K
    retrieval   : Recall@K and MRR against the corpus's own question_focus
                  labels - a chunk is relevant if it belongs to the document
                  the query was generated from
    grounding   : share of recommendations that retrieve any supporting chunk

The retrieval probe is self-supervised: queries are built from held-out chunk
metadata, so no hand-labelled relevance set is needed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate_retrieval(cfg: dict, n_queries: int = 100, seed: int = 42) -> dict:
    """Recall@K / MRR using each chunk's `question_focus` as ground truth.

    Two corpus quirks shape this probe:

    1. `section` is free text, not an enum. Some sections are entire disease
       names ("11q partial monosomy syndrome"), so naively concatenating
       focus + section produces a nonsense query about two unrelated diseases.
       We build the query from the focus and only append a section when it
       reads like a real heading.

    2. `document_id` is too strict a relevance label. The corpus holds
       near-duplicate documents for the same topic (0000579 and 0000580 are
       both "Human Papillomavirus (HPV) Vaccine"), so a retrieval that returns
       the sibling document is correct but would be scored wrong. We match on
       normalised `question_focus` instead, which is the actual topic identity.
    """
    from src.rag.retriever import EvidenceRetriever

    retriever = EvidenceRetriever(cfg)
    col = retriever.collection
    total = col.count()
    print(f"sampling {n_queries} probe queries from {total:,} chunks")

    def norm(text: str) -> str:
        return " ".join(str(text).lower().split())

    random.seed(seed)
    offsets = sorted(random.sample(range(total), min(n_queries * 4, total)))

    probes: list[tuple[str, str]] = []
    for off in offsets:
        got = col.get(limit=1, offset=off, include=["metadatas"])
        if not got["metadatas"]:
            continue
        meta = got["metadatas"][0]
        focus = meta.get("question_focus")
        section = str(meta.get("section", ""))
        if not focus:
            continue
        # Keep only short, heading-like sections; anything long is prose or a
        # disease name and would corrupt the query.
        query = str(focus)
        if section and len(section.split()) <= 3 and norm(section) not in norm(focus):
            query = f"{focus} {section.lower()}"
        probes.append((query, norm(focus)))
        if len(probes) >= n_queries:
            break

    k_values = (1, 3, 5)
    hits = {k: 0 for k in k_values}
    reciprocal = 0.0
    ungrounded = 0

    for i, (query, gold_focus) in enumerate(probes, 1):
        # No distance filter here: we are measuring ranking, not grounding.
        result = retriever.retrieve(query, top_k=max(k_values), max_distance=2.0)
        found = [norm(e.question_focus) for e in result.evidence]
        if not found:
            ungrounded += 1
            continue
        for k in k_values:
            if gold_focus in found[:k]:
                hits[k] += 1
        if gold_focus in found:
            reciprocal += 1.0 / (found.index(gold_focus) + 1)
        if i % 20 == 0:
            print(f"  {i}/{len(probes)}", end="\r")

    n = max(len(probes), 1)
    metrics = {"queries": len(probes), "empty_results": ungrounded,
               "mrr": round(reciprocal / n, 4)}
    for k in k_values:
        metrics[f"recall@{k}"] = round(hits[k] / n, 4)
    print(f"\nretrieval: {metrics}")
    return metrics


def evaluate_recommender(cfg: dict) -> dict:
    """Re-run the held-out test evaluation from the training script."""
    import pandas as pd
    import torch
    from collections import defaultdict
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module

    train_mod = import_module("03_train_recommender")
    from src.models.recommender import DrugRecommender

    rc = cfg["recommender"]
    out_dir = ROOT / rc["output_dir"]
    if not (out_dir / "pytorch_model.bin").exists():
        print("recommender not trained - skipping")
        return {}

    proc = ROOT / cfg["paths"]["processed"]
    meta = json.loads((out_dir / "label_maps.json").read_text(encoding="utf-8"))
    conditions, drugs = meta["conditions"], meta["drugs"]
    condition_drugs = defaultdict(set, {k: set(v) for k, v in
                                        meta.get("condition_drugs", {}).items()})

    test_df = pd.read_csv(proc / "recommender_test.csv")
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    dataset = train_mod.ReviewDataset(
        test_df, tokenizer, {c: i for i, c in enumerate(conditions)},
        {d: i for i, d in enumerate(drugs)}, meta["max_length"], condition_drugs,
    )
    loader = DataLoader(dataset, batch_size=rc["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DrugRecommender(meta["encoder"], len(conditions), len(drugs))
    model.load_state_dict(torch.load(out_dir / "pytorch_model.bin", map_location="cpu"))
    model.to(device)

    metrics = train_mod.evaluate(model, loader, device, drugs)
    print(f"recommender: {metrics}")
    return metrics


def evaluate_grounding(cfg: dict, n: int = 50) -> dict:
    """What share of conditions retrieve any supporting evidence?

    Expected to be well below 100%: the corpus is disease-information text and
    many common conditions (e.g. Birth Control) simply are not described in it.
    Reporting this honestly is the point.
    """
    from src.rag.retriever import EvidenceRetriever

    proc = ROOT / cfg["paths"]["processed"]
    path = proc / "conditions.json"
    if not path.exists():
        print("no conditions.json - run scripts/02_prepare_data.py first")
        return {}

    conditions = json.loads(path.read_text(encoding="utf-8"))[:n]
    retriever = EvidenceRetriever(cfg)
    grounded, per_condition = 0, {}
    for i, cond in enumerate(conditions, 1):
        result = retriever.retrieve_for_recommendation(cond, "")
        ok = result.is_grounded
        grounded += ok
        per_condition[cond] = {
            "grounded": ok,
            "top_relevance": result.evidence[0].relevance if ok else None,
        }
        if i % 10 == 0:
            print(f"  {i}/{len(conditions)}", end="\r")

    rate = round(grounded / max(len(conditions), 1), 4)
    print(f"\ngrounding: {grounded}/{len(conditions)} conditions "
          f"have supporting evidence ({rate:.1%})")
    return {"conditions_tested": len(conditions), "grounded": grounded,
            "grounding_rate": rate, "per_condition": per_condition}


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--recommender", action="store_true")
    ap.add_argument("--grounding", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-queries", type=int, default=100)
    args = ap.parse_args()

    if not any([args.retrieval, args.recommender, args.grounding, args.all]):
        ap.error("pick --retrieval, --recommender, --grounding, or --all")

    report: dict = {}
    if args.retrieval or args.all:
        report["retrieval"] = evaluate_retrieval(cfg, args.n_queries)
    if args.recommender or args.all:
        report["recommender"] = evaluate_recommender(cfg)
    if args.grounding or args.all:
        report["grounding"] = evaluate_grounding(cfg)

    out = ROOT / cfg["paths"]["artifacts"] / "evaluation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
