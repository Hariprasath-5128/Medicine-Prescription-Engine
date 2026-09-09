"""Run the whole pipeline within a wall-clock budget.

    python scripts/run_all.py --hours 24          # full day
    python scripts/run_all.py --hours 24 --dry-run

Stage order is deliberate. The cheap, high-certainty work runs first, so that a
budget overrun costs you the *optional* stage rather than the whole system:

    0  evidence DB      minutes    no GPU
    1  data prep        ~15 min    no GPU
    2  recommender      ~2-3 h     needed by the API and UI
    3  QDoRA QA         the rest   optional - the system works without it
    4  evaluation       ~20 min    always runs, even after a stage fails

Each stage receives the hours actually remaining, not its share of the original
budget, so time saved early is handed to the QA fine-tune.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, argv, needs_gpu, optional)
Stage = tuple[str, list[str], bool, bool]


def run(cmd: list[str], dry: bool) -> int:
    printable = " ".join(cmd)
    print(f"\n$ {printable}\n", flush=True)
    if dry:
        return 0
    # Stream output live: these stages run for hours and a silent terminal is
    # indistinguishable from a hang.
    return subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT).returncode


def fmt(hours: float) -> str:
    h = int(hours)
    return f"{h}h{int((hours - h) * 60):02d}m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24.0, help="total wall-clock budget")
    ap.add_argument("--recommender-hours", type=float, default=3.0)
    ap.add_argument("--skip-qa", action="store_true", help="skip the 8B QDoRA stage")
    ap.add_argument("--small", action="store_true", help="use Phi-3-mini for QA")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start = time.time()
    reserve_for_eval = 0.4  # hours held back so evaluation always gets to run

    stages: list[Stage] = [
        ("evidence DB", ["scripts/00_setup_evidence_db.py", "--verify"], False, False),
        ("data prep", ["scripts/02_prepare_data.py", "--all"], False, False),
    ]

    print("=" * 70)
    print(f"Medicine Prescription Engine - {fmt(args.hours)} budget")
    print("=" * 70)

    failed: list[str] = []
    for label, cmd, _needs_gpu, optional in stages:
        print(f"\n### {label}")
        if run(cmd, args.dry_run) != 0:
            failed.append(label)
            if not optional:
                print(f"!! required stage '{label}' failed - stopping")
                return 1

    elapsed = (time.time() - start) / 3600
    remaining = args.hours - elapsed - reserve_for_eval

    # Stage 2: recommender. Capped so it cannot eat the QA stage's time.
    rec_budget = min(args.recommender_hours, max(remaining, 0.5))
    print(f"\n### recommender  (elapsed {fmt(elapsed)}, "
          f"budget {fmt(rec_budget)} of {fmt(remaining)} remaining)")
    if run(["scripts/03_train_recommender.py", "--fp16",
            "--budget-hours", f"{rec_budget:.2f}"], args.dry_run) != 0:
        failed.append("recommender")
        print("!! recommender failed - the API and UI need it; continuing to evaluate")

    # Stage 3: QA fine-tune gets every hour the earlier stages did not use.
    if not args.skip_qa:
        elapsed = (time.time() - start) / 3600
        qa_budget = args.hours - elapsed - reserve_for_eval
        if qa_budget < 0.5:
            print(f"\n### QA fine-tune SKIPPED - only {fmt(max(qa_budget, 0))} left")
        else:
            print(f"\n### QA QDoRA  (elapsed {fmt(elapsed)}, budget {fmt(qa_budget)})")
            cmd = ["scripts/04_train_qa_dora.py", "--budget-hours", f"{qa_budget:.2f}"]
            if args.small:
                cmd.append("--small")
            if run(cmd, args.dry_run) != 0:
                failed.append("QA fine-tune")
                print("!! QA stage failed - the rest of the system still works")

    print("\n### evaluation")
    if run(["scripts/05_evaluate.py", "--all"], args.dry_run) != 0:
        failed.append("evaluation")

    total = (time.time() - start) / 3600
    print("\n" + "=" * 70)
    print(f"finished in {fmt(total)} of {fmt(args.hours)} budgeted")
    if failed:
        print(f"stages that failed: {', '.join(failed)}")
    else:
        print("all stages completed")
    print("\nnext:")
    print("  uvicorn src.api.main:app --port 8000")
    print("  streamlit run src/ui/app.py")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
