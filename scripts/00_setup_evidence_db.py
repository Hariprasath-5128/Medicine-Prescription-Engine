"""Copy the SERA Chroma index into this project as a self-contained evidence store.

SERA already holds 56,743 chunks embedded with BAAI/bge-m3, each carrying
`source_url`, `section`, `document_id` and `umls_semantic_group` metadata. That
is exactly the provenance chain this project needs, so we copy rather than
re-ingest (re-embedding 56k chunks would take hours on a 4 GB card).

    python scripts/00_setup_evidence_db.py --verify

Costs ~632 MB of disk. Use --link to skip the copy and point at SERA in place.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

SERA_CHROMA = Path("D:/Projects/Medicine-Prescription-Engine/data/chroma")
ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def copy_index(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        sys.exit(
            f"SERA index not found at {src}\n"
            "Point --source at your Chroma directory, or run SERA's ingestion first."
        )
    if dst.exists() and any(dst.iterdir()):
        if not force:
            print(f"{dst} already populated - skipping copy (use --force to overwrite)")
            return
        shutil.rmtree(dst)

    size_mb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e6
    print(f"copying {size_mb:,.0f} MB  {src} -> {dst}")
    # dirs_exist_ok lets us copy into the pre-created data/chroma placeholder.
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print("copy complete")


def verify(chroma_dir: Path, collection: str) -> None:
    """Confirm the copied index opens and still carries provenance metadata."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    names = [c.name for c in client.list_collections()]
    print(f"\ncollections: {names}")
    if collection not in names:
        sys.exit(f"expected collection '{collection}' missing from {chroma_dir}")

    col = client.get_collection(collection)
    print(f"'{collection}' holds {col.count():,} chunks")

    sample = col.get(limit=2, include=["metadatas", "documents"])
    for meta, doc in zip(sample["metadatas"], sample["documents"]):
        print("\n--- sample chunk ---")
        for key in ("document_id", "section", "question_focus",
                    "umls_semantic_group", "source_url"):
            if key in meta:
                print(f"  {key:22}: {meta[key]}")
        print(f"  text[:160]            : {doc[:160]}...")

    required = {"source_url", "section"}
    missing = required - set(sample["metadatas"][0])
    if missing:
        print(f"\nWARNING: chunks lack {missing}; citations will be incomplete.")
    else:
        print("\nprovenance metadata present - citations will resolve to source URLs")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SERA_CHROMA)
    ap.add_argument("--force", action="store_true", help="overwrite an existing index")
    ap.add_argument("--verify", action="store_true", help="open the index and print samples")
    ap.add_argument("--link", action="store_true",
                    help="do not copy; just verify the source in place")
    args = ap.parse_args()

    dst = ROOT / cfg["paths"]["chroma"]
    target = args.source if args.link else dst

    if not args.link:
        copy_index(args.source, dst, args.force)
    if args.verify or args.link:
        verify(target, cfg["evidence"]["collection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
