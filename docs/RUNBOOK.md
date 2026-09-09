# Runbook - what to do at each step

Every command runs from the project root:

```bash
cd "C:/Projects/Medicine Prescription Engine"
```

**Shortcut:** `python scripts/run_all.py --hours 24` does steps 0-5 in order and
hands each stage the time actually remaining. Use the per-step guide below when
you want control, or when a stage fails and you need to rerun just that one.

---

## Step 0 - Copy the evidence index

```bash
python scripts/00_setup_evidence_db.py --verify
```

**What it does.** Copies `C:/Projects/SERA/data/chroma` (~632 MB) into
`data/chroma`, then opens it and prints two sample chunks with their metadata.

**Why.** These 56,407 chunks are already embedded with `BAAI/bge-m3` and carry
`source_url`, `section` and `document_id`. That metadata *is* the citation
layer - re-embedding from scratch would take hours and gain nothing.

**Success looks like:**

```
'raw_chunks' holds 56,407 chunks
provenance metadata present - citations will resolve to source URLs
```

**If it fails.** `SERA index not found` means the path moved - pass
`--source <path>`. To save 632 MB, use `--link` to read SERA in place instead
of copying (but then SERA must stay where it is).

**Run again?** No. One-time. Add `--force` only to re-copy over an existing index.

---

## Step 1 - Audit a dataset (optional, but do it before trusting any new file)

```bash
python scripts/01_audit_dataset.py data/raw/<file>.csv --text-col review
```

**What it does.** Reports distinct values per column, duplicate rows ignoring
any ID column, and whether free text is real language or a fixed phrase list.
Exits non-zero when the data is unfit for supervised learning.

**Why this exists.** The Kaggle file `medical-conditions50000` advertises 50,000
rows. It has **20**, copy-pasted 2,500 times each - 49,980 exact duplicates.
Any random split puts identical rows on both sides, so accuracy would read
~100% while measuring nothing. This script catches that class of problem in
seconds:

```
distinct content rows : 20 (0.0%)
VERDICT: UNFIT for supervised training
```

Contrast UltraMedical, which passes at 96.07% distinct.

**Run again?** Only when you introduce a new dataset.

---

## Step 2 - Build the training data

```bash
python scripts/02_prepare_data.py --all
```

**What it does.** Two independent jobs:

*Recommender* - downloads the UCI Drug Review set from Kaggle, strips HTML
entities, drops junk conditions, **removes 47,587 duplicate reviews**, then
splits stratified by condition. It asserts no review text appears in two splits
and crashes if one does.

*QA* - downloads UltraMedical (409,593 rows), keeps the Open-End and Literature
types, drops ~30 degenerate rows, dedups, and writes JSONL.

**Why the dedup is not optional.** Splitting before removing duplicates
recreates exactly the leakage that disqualifies the 50k file.

**Success looks like:**

```
after dedup        : 110,236  (removed 47,587 copies)
split train/val/test = 81,137/7,159/7,160
leakage check passed: no review text shared across splits
...
kept 181,700 pairs after type filter ['Literature', 'Open-End'] and dedup
split train/val = 178,066/3,634  (no shared questions)
```

**Cost.** ~15 min, no GPU. Writes ~575 MB - `qa_train.jsonl` alone is 523 MB.

**If it fails.** Kaggle 403 means `~/.kaggle/kaggle.json` is missing or the
dataset rules are unaccepted - open the dataset page once in a browser. Use
`--recommender` or `--qa` to rebuild just one half.

---

## Step 3 - Train the recommender

```bash
python scripts/03_train_recommender.py --fp16              # 12 GB box
python scripts/03_train_recommender.py --batch-size 4 --fp16   # 4 GB card
python scripts/03_train_recommender.py --limit 2000 --epochs 1 --fp16  # smoke test
```

**What it does.** Fine-tunes BioLinkBERT-large (335.7M) with two heads over one
shared encoder: a softmax condition head (90 classes) and a **sigmoid
multi-label drug head** (2,044 drugs).

**Why multi-label.** A condition legitimately has several valid drugs. Forcing
one "correct" answer would be clinically wrong and would inflate the metric;
multi-label is also what makes Precision@K / Recall@K meaningful.

**What happens first.** Before training, it times real optimiser steps on your
GPU and prints:

```
--- recommender throughput (measured on this GPU) ---
  0.355 s/step at batch 16 = 45.0 samples/s
  full dataset: 81,137 rows x 3 epochs = 1.5 h
  -> FITS. Training on all 81,137 rows.
```

If the full run would exceed `budget_hours` (default 3), it subsamples instead
of overrunning. It never trusts an estimate taken from another GPU - doing that
once produced a figure that was wrong by ~5x.

**Success looks like** rising val macro-F1 across epochs, `saved new best`, and
a final test block written to `artifacts/recommender/test_metrics.json`.

**Judging the numbers.** 90 imbalanced classes; random is ~1%. Condition
macro-F1 in the 0.3-0.5 range is a reasonable result on noisy patient prose -
this data is a proxy for prescribing, not a record of it. **Macro-F1 is the
metric to watch, not accuracy**, because "Birth Control" alone is ~30% of rows.

**If it fails.** CUDA OOM -> lower `--batch-size` (6.71 GB measured at 8) or use
`--encoder base`. Very slow steps -> check `nvidia-smi`; near-full VRAM means
the driver is spilling to system RAM.

**Required by** the API and Streamlit UI. Without it they run in evidence-search
mode only.

---

## Step 4 - DoRA fine-tune the QA model

```bash
python scripts/04_train_qa_dora.py --check     # ALWAYS run this first
python scripts/04_train_qa_dora.py             # Llama-3-8B-UltraMedical
python scripts/04_train_qa_dora.py --small     # Phi-3-mini fallback
```

**Run `--check` first.** It verifies VRAM, DoRA support, and - critically - that
your torch build matches the GPU. The RTX 5070 is Blackwell (sm_120) and needs
a **cu128** wheel; a cu121 wheel silently falls back to CPU, which turns a 7-hour
run into a week:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**What it does.** QDoRA - the 8B base is frozen at 4-bit NF4, and only DoRA
adapters train in bf16, so quantisation noise never enters the gradients.
r=32/alpha=64 with NEFTune (alpha 5) for accuracy; seq 1024, batch 1 x accum 16.

**Why seq 1024, not 2048.** Published 8B runs at seq 2048 peak at 14-15 GB and
OOM a 12 GB card. 1024 keeps peak near 8 GB.

**Success looks like** `trainable params: ~84M || trainable%: ~1.0`, a falling
loss, then `adapter saved`. Only the adapter is written (~300 MB), not the base.

**Cost.** The largest stage. It measures throughput first and trims the dataset
to `--budget-hours` (default 14). A useful reference point: an RTX 3090 does 8B
LoRA over 52k samples x 3 epochs in ~78 min, and QDoRA is ~15% slower - but
**trust the printed measurement, not this note**.

**This stage is optional.** Skip it and everything else still works; the
recommender plus RAG is the functioning system.

**If it fails.** OOM -> `--small`. Killed silently -> usually the Xet CDN on
Windows; the scripts already set `HF_HUB_DISABLE_XET=1`. Want a short run ->
`--limit 20000`.

---

## Step 5 - Evaluate

```bash
python scripts/05_evaluate.py --all
python scripts/05_evaluate.py --retrieval      # no trained model needed
```

**What it does.** Three independent measurements, written to
`artifacts/evaluation_report.json`:

- **retrieval** - Recall@K and MRR using each chunk's `question_focus` as ground
  truth. Self-supervised, so no hand-labelled relevance set is needed.
- **recommender** - condition accuracy/macro-F1 and drug Precision@K/Recall@K on
  the held-out test split.
- **grounding** - what share of conditions retrieve any supporting evidence.

**Baseline already measured** (retrieval works without any training):

```
MRR 0.4881 | recall@1 0.4667 | recall@3 0.5000 | recall@5 0.5333
```

Recall rising with K is the sanity check. If recall@1 == recall@5 exactly,
the probe is broken, not brilliant - that happened once and was a bug.

**Expect grounding below 100%.** The corpus is disease-information text; many
conditions (Birth Control, Weight Loss) simply are not described in it.
Reporting that honestly is the point.

---

## Steps 6/7 - Serve

```bash
uvicorn src.api.main:app --reload --port 8000   # http://127.0.0.1:8000/docs
streamlit run src/ui/app.py                     # http://localhost:8501
```

**Check readiness first:** `curl http://127.0.0.1:8000/health` reports whether
the evidence DB and the trained recommender are present.

**API.** `POST /recommend` runs the full pipeline; `POST /evidence` is retrieval
only; `GET /evidence/{chunk_id}` returns one chunk with full provenance.

**UI.** Three tabs. *Recommendation* needs step 3. *Evidence search* works as
soon as step 0 is done. Each citation expands to show the retrieved chunk, its
section, relevance, and a link to the original source.

**Both are lazy-loading** - the server starts even when the recommender is
untrained, and the UI degrades to search-only rather than crashing.

---

## Suggested one-day order

| # | Stage | Time | GPU | Skippable |
|---|---|---|---|---|
| 0 | Evidence DB | minutes | no | no |
| 2 | Data prep | ~15 min | no | no |
| 3 | Recommender | ~2-3 h | yes | no |
| 4 | QA QDoRA | the remainder | yes | **yes** |
| 5 | Evaluation | ~20 min | partly | no |

Do a smoke test of step 3 (`--limit 2000 --epochs 1 --fp16`) before the real run.
Ten minutes spent there is cheaper than discovering a config problem eight hours
into step 4.

## A note on the current artifacts

`artifacts/` was cleared deliberately. It held a recommender from a 600-row
smoke test (macro-F1 0.0035) and a QA adapter from an 8-row test with an
unrelated 135M model. Those numbers measure nothing, and leaving them on disk
invites mistaking them for results. Steps 3 and 4 regenerate them properly.
