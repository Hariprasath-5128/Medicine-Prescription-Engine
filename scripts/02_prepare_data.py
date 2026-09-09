"""Download and clean the real training corpora.

Two datasets, each doing a job the 50k template file could not:

  drug_reviews  UCI Drug Review (Kaggle: jessicali9530/kuc-hackathon-winter-2018)
                161,297 train rows, 3,436 distinct drugs, 884 conditions,
                112,329 distinct free-text reviews written by patients.
                -> trains the condition -> drug recommender.

  qa_pairs      Built from the SERA chunk corpus (MedQuAD-style question_focus +
                section text) into instruction pairs.
                -> supervised data for the DoRA QA fine-tune.

    python scripts/02_prepare_data.py --all

Deduplication is not optional. The drug-review file contains 48,968 exact
duplicate reviews; splitting before dropping them leaks test rows into train,
which is the same flaw that makes the 50k dataset unusable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DRUG_REVIEW_REF = "jessicali9530/kuc-hackathon-winter-2018"


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _clean_review(text: str) -> str:
    """Strip the HTML entities and wrapping quotes the UCI export is full of."""
    text = str(text)
    for bad, good in (("&#039;", "'"), ("&amp;", "&"), ("&quot;", '"'),
                      ("&lt;", "<"), ("&gt;", ">"), ("&rsquo;", "'")):
        text = text.replace(bad, good)
    text = text.strip().strip('"').strip()
    return re.sub(r"\s+", " ", text)


def fetch_drug_reviews(raw_dir: Path) -> Path:
    target = raw_dir / "drugsComTrain_raw.csv"
    if target.exists():
        print(f"already present: {target.name}")
        return target
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"downloading {DRUG_REVIEW_REF} ...")
    api.dataset_download_files(DRUG_REVIEW_REF, path=str(raw_dir), unzip=True, quiet=False)
    if not target.exists():
        sys.exit(f"expected {target} after download; check the Kaggle dataset layout")
    return target


def build_recommender_data(cfg: dict) -> None:
    raw_dir = ROOT / cfg["paths"]["raw"]
    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = cfg["recommender"]

    src = fetch_drug_reviews(raw_dir)
    df = pd.read_csv(src)
    print(f"\nloaded {len(df):,} rows")

    df = df.dropna(subset=["condition", "drugName", "review"])
    # The scrape left rows whose condition is literally "3</span> users found
    # this comment helpful." - drop that whole class of junk.
    df = df[~df["condition"].str.contains("</span>", na=False)]
    df["review"] = df["review"].map(_clean_review)
    df = df[df["review"].str.split().str.len() >= 5]
    print(f"after cleaning     : {len(df):,}")

    if rc["drop_duplicate_text"]:
        before = len(df)
        df = df.drop_duplicates(subset=["review"])
        print(f"after dedup        : {len(df):,}  (removed {before - len(df):,} copies)")

    counts = df["condition"].value_counts()
    keep = counts[counts >= rc["min_condition_count"]].index
    df = df[df["condition"].isin(keep)]
    print(f"after rare-class cut: {len(df):,}  ({df['condition'].nunique()} conditions)")

    # Stratified split so every condition appears in train and test, and so the
    # test set is a fair sample rather than a handful of dominant classes.
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["condition"]
    )
    val, test = train_test_split(
        test, test_size=0.5, random_state=42, stratify=test["condition"]
    )

    overlap = set(train["review"]) & set(test["review"])
    assert not overlap, f"LEAKAGE: {len(overlap)} reviews in both train and test"
    print(f"\nsplit train/val/test = {len(train):,}/{len(val):,}/{len(test):,}")
    print("leakage check passed: no review text shared across splits")

    cols = ["drugName", "condition", "review", "rating"]
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"recommender_{name}.csv"
        part[cols].to_csv(path, index=False)
        print(f"  wrote {path.relative_to(ROOT)}")

    labels = sorted(df["condition"].unique())
    (out_dir / "conditions.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    drugs = sorted(df["drugName"].unique())
    (out_dir / "drugs.json").write_text(json.dumps(drugs, indent=2), encoding="utf-8")
    print(f"  wrote {len(labels)} conditions, {len(drugs)} drugs")


def build_qa_data(cfg: dict) -> None:
    """Build instruction pairs from UltraMedical.

    Why an external dataset rather than the SERA chunks: fine-tuning on the same
    text the RAG index already holds is self-distillation. It teaches format,
    adds no knowledge, and encourages the model to recite corpus text from
    memory - uncited - which is exactly what the citation layer exists to
    prevent. UltraMedical keeps the two roles separate: it supplies instruction
    behaviour, SERA supplies verifiable evidence.

    UltraMedical (NeurIPS 2024 D&B Spotlight) aggregates MedQA, MedMCQA,
    PubMedQA, TextBookQA, ChatDoctor, MedInstruct and MedQuAD into 409,593 rows,
    audited here at 96.07% distinct questions.
    """
    from datasets import load_dataset

    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dc = cfg["qa_model"]["dataset"]

    print(f"loading {dc['name']} ...")
    ds = load_dataset(dc["name"], split=dc["split"])
    print(f"  {len(ds):,} rows")

    keep_types = set(dc["keep_types"])
    min_q, min_a = dc["min_question_words"], dc["min_answer_words"]

    rows: list[dict] = []
    for row in ds:
        if keep_types and row["type"] not in keep_types:
            continue
        convo = row["conversations"]
        if len(convo) < 2:
            continue
        question = str(convo[0]["value"]).strip()
        answer = str(convo[1]["value"]).strip()
        # Guards against the ~30 degenerate rows whose answer is "No" or a bare
        # option letter - they teach the model to be terse, not to explain.
        if len(question.split()) < min_q or len(answer.split()) < min_a:
            continue
        rows.append({
            "instruction": question,
            "output": answer,
            "type": row["type"],
            # e.g. "PubMedQA,1234" -> "PubMedQA", so a split can be traced.
            "source": str(row["id"]).split(",")[0],
        })

    qa = pd.DataFrame(rows).drop_duplicates(subset=["instruction"])
    print(f"kept {len(qa):,} pairs after type filter {sorted(keep_types)} and dedup")

    if dc["max_rows"]:
        qa = qa.sample(n=min(dc["max_rows"], len(qa)), random_state=42)
        print(f"subsampled to {len(qa):,} (config max_rows)")

    print("\ncomposition:")
    for source, count in qa["source"].value_counts().items():
        print(f"  {source:<26}{count:>8,}")

    qa = qa.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cut = int(len(qa) * (1 - dc["val_fraction"]))
    train, val = qa.iloc[:cut], qa.iloc[cut:]

    overlap = set(train["instruction"]) & set(val["instruction"])
    assert not overlap, f"LEAKAGE: {len(overlap)} questions in both train and val"
    print(f"\nsplit train/val = {len(train):,}/{len(val):,}  (no shared questions)")

    # Write JSONL by hand rather than via DataFrame.to_json: medical answers
    # contain literal newlines, and pandas does not escape them inside the
    # emitted strings, so a reader splitting on "\n" hits truncated JSON.
    for name, part in (("train", train), ("val", val)):
        path = out_dir / f"qa_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for record in part.to_dict(orient="records"):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  wrote {path.relative_to(ROOT)}")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recommender", action="store_true")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not (args.recommender or args.qa or args.all):
        ap.error("pick --recommender, --qa, or --all")
    if args.recommender or args.all:
        build_recommender_data(cfg)
    if args.qa or args.all:
        build_qa_data(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
