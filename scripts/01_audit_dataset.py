"""Audit a candidate dataset for *information content*, not row count.

The motivating case: `abdelrahmangamal236/medical-conditions50000` advertises
50,000 rows. It has 20. Every column holds exactly 20 distinct values and the
remaining 49,980 rows are verbatim copies. A model trained on it cannot help
but score ~100% on any random split, because the test rows are literally
present in the train split.

Run this before trusting *any* dataset:

    python scripts/01_audit_dataset.py data/raw/foo.csv --text-col review

Exit code is 1 when the data is unfit for supervised learning, so this can gate
a training run in CI or a Makefile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Below this ratio of distinct-rows-to-total, a random split leaks by construction.
DUPLICATION_FAIL_RATIO = 0.10
# A free-text column with fewer distinct values than this is a template, not language.
TEMPLATE_VALUE_CEILING = 100


def _fmt(n: int, total: int) -> str:
    return f"{n:,} ({n / total:.1%})" if total else f"{n:,}"


def audit(path: Path, text_col: str | None, label_col: str | None) -> bool:
    df = pd.read_csv(path, on_bad_lines="skip")
    total = len(df)
    print(f"\n=== {path.name} ===")
    print(f"rows={total:,}  cols={len(df.columns)}")
    print(f"columns: {list(df.columns)}\n")

    # Per-column cardinality. The tell-tale sign of a templated dataset is a
    # tiny nunique across *every* column at once.
    print(f"{'column':<32}{'distinct':>12}{'nulls':>10}  verdict")
    print("-" * 78)
    templated: list[str] = []
    for col in df.columns:
        n_uniq = df[col].nunique(dropna=True)
        nulls = int(df[col].isna().sum())
        verdict = ""
        # An ID column is legitimately all-distinct; ignore it.
        is_id_like = n_uniq == total
        if not is_id_like and n_uniq <= TEMPLATE_VALUE_CEILING and total > 1_000:
            ratio = total / max(n_uniq, 1)
            verdict = f"TEMPLATED (~{ratio:,.0f} copies each)"
            templated.append(col)
        print(f"{str(col):<32}{n_uniq:>12,}{nulls:>10,}  {verdict}")

    # Duplication ignoring any surrogate key, which is what a split actually sees.
    id_like = [c for c in df.columns if df[c].nunique() == total]
    content_cols = [c for c in df.columns if c not in id_like]
    distinct_rows = len(df[content_cols].drop_duplicates()) if content_cols else total
    dup_rows = total - distinct_rows

    print(f"\nignoring key column(s) {id_like or 'none'}:")
    print(f"  distinct content rows : {_fmt(distinct_rows, total)}")
    print(f"  duplicate rows        : {_fmt(dup_rows, total)}")

    problems: list[str] = []
    if total and distinct_rows / total < DUPLICATION_FAIL_RATIO:
        problems.append(
            f"only {distinct_rows:,} distinct rows behind {total:,} advertised "
            f"({distinct_rows / total:.2%}). A random train/test split puts identical "
            f"rows on both sides, so reported accuracy is leakage, not skill."
        )
    if len(templated) >= max(2, len(content_cols) - 1):
        problems.append(
            f"every meaningful column is templated ({', '.join(templated)}). "
            f"This is a lookup table; a dict reproduces it exactly."
        )

    # Free-text realism: real prose has varied length. Templates do not.
    if text_col and text_col in df.columns:
        lens = df[text_col].astype(str).str.split().str.len()
        n_uniq = df[text_col].nunique()
        print(f"\ntext column '{text_col}':")
        print(f"  distinct values : {n_uniq:,}")
        print(f"  words  min/med/max : {lens.min()}/{int(lens.median())}/{lens.max()}")
        if n_uniq <= TEMPLATE_VALUE_CEILING:
            problems.append(
                f"'{text_col}' has {n_uniq} distinct strings - it is a fixed phrase "
                f"list, so no NLP model can learn language from it."
            )

    if label_col and label_col in df.columns:
        vc = df[label_col].value_counts()
        print(f"\nlabel '{label_col}': {len(vc)} classes, "
              f"imbalance {vc.max() / max(vc.min(), 1):.1f}x (max/min)")
        print(vc.head(10).to_string())

    print()
    if problems:
        print("VERDICT: UNFIT for supervised training")
        for p in problems:
            print(f"  - {p}")
        return False
    print("VERDICT: usable - variation is real, no split-level duplication")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+", help="dataset file(s) to audit")
    ap.add_argument("--text-col", help="free-text column to check for real language")
    ap.add_argument("--label-col", help="target column to summarise")
    args = ap.parse_args()

    ok = True
    for path in args.csv:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            ok = False
            continue
        ok &= audit(path, args.text_col, args.label_col)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
