"""Prepare one laptop for its half of a two-machine training split.

    python scripts/setup_machine.py --role recommender   # laptop A
    python scripts/setup_machine.py --role qa            # laptop B

The two training stages share no files, so they parallelise cleanly:

    laptop A  recommender  needs recommender_*.csv + label maps    47 MB
    laptop B  QA QDoRA     needs qa_train.jsonl + qa_val.jsonl    534 MB

This script fetches only the LFS objects that role actually needs, checks the
GPU, and reports what is missing. Pulling all 581 MB on both machines wastes
GitHub LFS bandwidth (free tier: 1 GB/month) and time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROLES = {
    "recommender": {
        "patterns": [
            "data/processed/recommender_train.csv",
            "data/processed/recommender_val.csv",
            "data/processed/recommender_test.csv",
        ],
        "plain": ["data/processed/conditions.json", "data/processed/drugs.json"],
        "mb": 47,
        "command": "python scripts/03_train_recommender.py --fp16",
        "needs_chroma": False,
    },
    "qa": {
        "patterns": ["data/processed/qa_train.jsonl", "data/processed/qa_val.jsonl"],
        "plain": [],
        "mb": 534,
        "command": "python scripts/04_train_qa_dora.py",
        "needs_chroma": False,
    },
    # Whichever machine finishes first collects both artifacts and evaluates.
    "eval": {
        "patterns": [
            "data/processed/recommender_test.csv",
            "data/processed/qa_val.jsonl",
        ],
        "plain": ["data/processed/conditions.json", "data/processed/drugs.json"],
        "mb": 15,
        "command": "python scripts/05_evaluate.py --all",
        "needs_chroma": True,
    },
}


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def check_gpu() -> bool:
    """Report the GPU and flag the Blackwell/cu128 trap before a long run."""
    try:
        import torch
    except ImportError:
        print("  torch not installed")
        return False

    if not torch.cuda.is_available():
        print("  no CUDA device visible")
        return False

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    cuda_ver = torch.version.cuda or "0.0"
    print(f"  GPU: {props.name} | {vram:.1f} GB | sm_{props.major}{props.minor} "
          f"| torch {torch.__version__} (cu{cuda_ver})")

    ok = True
    # A cu121 wheel on sm_120 does not error - it silently runs on CPU, which
    # turns a 7-hour job into a week.
    if props.major >= 12 and tuple(int(x) for x in cuda_ver.split(".")[:2]) < (12, 8):
        print("  !! Blackwell (sm_120) needs a cu128 build, or it falls back to CPU:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu128")
        ok = False
    if vram < 11:
        print(f"  !! {vram:.1f} GB is tight for the 8B QDoRA stage; consider --small")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, choices=sorted(ROLES))
    ap.add_argument("--skip-pull", action="store_true", help="only run the checks")
    args = ap.parse_args()

    role = ROLES[args.role]
    print(f"=== preparing this machine for role: {args.role} ===\n")

    print("GPU")
    gpu_ok = check_gpu()

    print(f"\ndata (~{role['mb']} MB for this role)")
    if args.skip_pull:
        print("  --skip-pull: not fetching")
    else:
        for pattern in role["patterns"]:
            print(f"  fetching {pattern} ...", flush=True)
            # LFS resumes, so a dropped transfer can simply be re-run.
            code, out = sh(["git", "lfs", "pull", "--include", pattern])
            if code != 0:
                print(f"    FAILED: {out.splitlines()[-1] if out else code}")
                print(f"    retry:  git lfs pull --include=\"{pattern}\"")

    print("\nverifying")
    missing: list[str] = []
    for path in role["patterns"] + role["plain"]:
        f = ROOT / path
        if not f.exists():
            missing.append(path)
            print(f"  MISSING  {path}")
            continue
        size = f.stat().st_size
        # An unfetched LFS file is a ~130-byte text pointer, not the real data.
        if size < 1000 and path.endswith((".csv", ".jsonl")):
            missing.append(path)
            print(f"  POINTER  {path} ({size} B - LFS content not fetched)")
        else:
            print(f"  ok       {path}  {size / 1e6:.1f} MB")

    if role["needs_chroma"]:
        chroma = ROOT / "data" / "chroma"
        if chroma.exists() and any(chroma.iterdir()):
            print("  ok       data/chroma")
        else:
            missing.append("data/chroma")
            print("  MISSING  data/chroma - run scripts/00_setup_evidence_db.py")

    print("\n" + "=" * 62)
    if missing or not gpu_ok:
        print("NOT READY")
        for m in missing:
            print(f"  missing: {m}")
        if not gpu_ok:
            print("  GPU/torch needs attention (see above)")
        print("\nIf a large LFS pull keeps dropping, regenerate instead:")
        print("  python scripts/02_prepare_data.py --all")
        return 1

    print("READY. Start training with:")
    print(f"  {role['command']}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
