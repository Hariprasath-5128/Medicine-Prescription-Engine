"""Fine-tune the condition/drug recommender.

    python scripts/03_train_recommender.py                      # BioLinkBERT-large
    python scripts/03_train_recommender.py --encoder base       # 4 GB card
    python scripts/03_train_recommender.py --limit 2000 --fp16  # smoke test

Default encoder is BioLinkBERT-large (333M), which is SOTA on BLURB and needs
~9 GB - run it on the 12 GB box. Pass `--encoder base` (110M, ~3 GB) to train
on a 4 GB card, or `--encoder biomedbert` for the previous default.

Gradient accumulation keeps the effective batch at 16 even when the physical
batch is 8, so the large encoder trains with small-batch memory.

Evaluation reports condition accuracy and macro-F1 plus drug Precision@K /
Recall@K on a test split that shares no review text with train (enforced in
scripts/02_prepare_data.py).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# The Xet CDN backend fails to download on some Windows setups, and it fails
# *silently* - the process dies with no traceback. Falling back to the classic
# resolver is reliable. Set HF_HUB_DISABLE_XET=0 to opt back in.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.recommender import DrugRecommender  # noqa: E402


class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, condition_ids: dict[str, int],
                 drug_ids: dict[str, int], max_length: int,
                 condition_drugs: dict[str, set[str]]):
        self.texts = df["review"].tolist()
        self.conditions = df["condition"].tolist()
        self.tokenizer = tokenizer
        self.condition_ids = condition_ids
        self.drug_ids = drug_ids
        self.max_length = max_length
        self.condition_drugs = condition_drugs

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        condition = self.conditions[i]

        # Supervise the drug head with every drug attested for this condition,
        # not just the one in this row: a review mentioning Sertraline for
        # Depression does not make Fluoxetine a negative example.
        drug_target = torch.zeros(len(self.drug_ids))
        for drug in self.condition_drugs.get(condition, ()):
            drug_target[self.drug_ids[drug]] = 1.0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "condition_labels": torch.tensor(self.condition_ids[condition]),
            "drug_labels": drug_target,
        }


@torch.no_grad()
def evaluate(model, loader, device, drugs: list[str], k_values=(1, 3, 5)) -> dict:
    model.eval()
    correct = total = 0
    preds, golds = [], []
    hits = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"])

        pred = out["condition_logits"].argmax(-1)
        gold = batch["condition_labels"]
        correct += (pred == gold).sum().item()
        total += len(gold)
        preds.extend(pred.cpu().tolist())
        golds.extend(gold.cpu().tolist())

        drug_scores = torch.sigmoid(out["drug_logits"])
        for row, truth in zip(drug_scores, batch["drug_labels"]):
            relevant = set(torch.nonzero(truth).flatten().cpu().tolist())
            if not relevant:
                continue
            for k in k_values:
                topk = set(row.topk(min(k, len(drugs))).indices.cpu().tolist())
                inter = len(topk & relevant)
                hits[k].append(inter / k)
                recalls[k].append(inter / len(relevant))

    from sklearn.metrics import f1_score

    metrics = {
        "condition_accuracy": round(correct / max(total, 1), 4),
        "condition_macro_f1": round(f1_score(golds, preds, average="macro",
                                             zero_division=0), 4),
    }
    for k in k_values:
        metrics[f"drug_precision@{k}"] = round(float(np.mean(hits[k])), 4) if hits[k] else 0.0
        metrics[f"drug_recall@{k}"] = round(float(np.mean(recalls[k])), 4) if recalls[k] else 0.0
    return metrics


def main() -> int:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    rc = cfg["recommender"]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="cap training rows for a smoke test")
    ap.add_argument("--epochs", type=int, default=rc["epochs"])
    ap.add_argument("--batch-size", type=int, default=rc["batch_size"])
    ap.add_argument("--fp16", action="store_true", help="mixed precision, saves VRAM")
    ap.add_argument("--encoder", help="model id, or a key from encoder_options "
                                      "(large / base / biomedbert)")
    ap.add_argument("--budget-hours", type=float, default=rc.get("budget_hours", 0),
                    help="wall-clock ceiling; 0 disables subsampling")
    args = ap.parse_args()

    # Let --encoder take either a shorthand key or a full HF model id.
    encoder = rc["encoder"]
    if args.encoder:
        encoder = rc.get("encoder_options", {}).get(args.encoder, args.encoder)

    proc = ROOT / cfg["paths"]["processed"]
    if not (proc / "recommender_train.csv").exists():
        raise SystemExit("run: python scripts/02_prepare_data.py --recommender")

    train_df = pd.read_csv(proc / "recommender_train.csv")
    val_df = pd.read_csv(proc / "recommender_val.csv")
    test_df = pd.read_csv(proc / "recommender_test.csv")
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(args.limit // 5, len(val_df)), random_state=42)

    conditions = json.loads((proc / "conditions.json").read_text(encoding="utf-8"))
    drugs = json.loads((proc / "drugs.json").read_text(encoding="utf-8"))
    condition_ids = {c: i for i, c in enumerate(conditions)}
    drug_ids = {d: i for i, d in enumerate(drugs)}

    # Which drugs are actually attested for each condition (from train only,
    # so the val/test drug targets carry no information from their own rows).
    condition_drugs: dict[str, set[str]] = defaultdict(set)
    for cond, drug in zip(train_df["condition"], train_df["drugName"]):
        if drug in drug_ids:
            condition_drugs[cond].add(drug)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"encoder={encoder}")
    print(f"device={device} | train={len(train_df):,} val={len(val_df):,} "
          f"test={len(test_df):,} | {len(conditions)} conditions, {len(drugs)} drugs")

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    make = lambda df: ReviewDataset(df, tokenizer, condition_ids, drug_ids,  # noqa: E731
                                    rc["max_length"], condition_drugs)
    train_loader = DataLoader(make(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(make(val_df), batch_size=args.batch_size)
    test_loader = DataLoader(make(test_df), batch_size=args.batch_size)

    model = DrugRecommender(encoder, len(conditions), len(drugs)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rc["learning_rate"]),
                                  weight_decay=rc.get("weight_decay", 0.01))

    # Time real steps on THIS gpu rather than trusting an estimate, then trim
    # the dataset if the full run would blow the wall-clock budget.
    if args.budget_hours and device == "cuda" and not args.limit:
        from src.utils.budget import measure, plan

        probe = next(iter(train_loader))
        probe = {k: v.to(device) for k, v in probe.items()}
        scaler_probe = torch.cuda.amp.GradScaler(enabled=args.fp16)

        def _step():
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler_probe.is_enabled()):
                loss = model(**probe)["loss"]
            scaler_probe.scale(loss).backward()
            scaler_probe.step(optimizer)
            scaler_probe.update()

        tp = measure(_step, args.batch_size)
        rows, report = plan(tp, len(train_df), args.budget_hours,
                            args.epochs, "recommender")
        print(report)

        if rows < len(train_df):
            train_df = train_df.sample(n=rows, random_state=42)
            train_loader = DataLoader(make(train_df), batch_size=args.batch_size,
                                      shuffle=True)
        # The probe left optimiser state and a warm scheduler behind; rebuild
        # both so training starts from the intended initial conditions.
        model = DrugRecommender(encoder, len(conditions), len(drugs)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(rc["learning_rate"]),
                                      weight_decay=rc.get("weight_decay", 0.01))
    accum = rc.get("gradient_accumulation_steps", 1)
    steps = (len(train_loader) // accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * rc.get("warmup_ratio", 0.06)), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device == "cuda")

    out_dir = ROOT / rc["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    patience = rc.get("early_stopping_patience", 0)
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                # Scale so accumulated gradients average rather than sum.
                loss = model(**batch)["loss"] / accum
            scaler.scale(loss).backward()

            # Step only once per `accum` micro-batches: this is what lets a
            # 340M encoder train at batch 8 while behaving like batch 16.
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.item() * accum
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / step:.4f}", end="\r")

        metrics = evaluate(model, val_loader, device, drugs)
        print(f"\nepoch {epoch} val: {metrics}")

        if metrics["condition_macro_f1"] > best_f1:
            best_f1 = metrics["condition_macro_f1"]
            torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(out_dir)
            (out_dir / "label_maps.json").write_text(json.dumps({
                "encoder": encoder,
                "conditions": conditions,
                "drugs": drugs,
                "max_length": rc["max_length"],
                "condition_drugs": {k: sorted(v) for k, v in condition_drugs.items()},
            }, indent=2), encoding="utf-8")
            print(f"  saved new best (macro-F1 {best_f1:.4f}) -> {out_dir}")
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                print(f"  no val gain for {stale} epochs - early stopping")
                break

    print("\nfinal test-set evaluation (no text shared with train):")
    test_metrics = evaluate(model, test_loader, device, drugs)
    for key, value in test_metrics.items():
        print(f"  {key:24}: {value}")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2),
                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
