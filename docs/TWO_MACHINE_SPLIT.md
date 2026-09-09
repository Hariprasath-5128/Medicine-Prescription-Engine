# Training across two RTX 5070 laptops

The two training stages read **no files in common**, so they run in parallel
with no coordination during training:

| | reads | writes |
|---|---|---|
| `03_train_recommender.py` | `recommender_*.csv`, `conditions.json`, `drugs.json` | `artifacts/recommender/` |
| `04_train_qa_dora.py` | `qa_train.jsonl` | `artifacts/qa-dora/` |

Wall-clock drops from *recommender + QA* to *max(recommender, QA)* - roughly
10 h to roughly 7 h, and the QA stage gets the whole day instead of a share
of it.

## Assignment

**Laptop A -> recommender. Laptop B -> QA fine-tune.**

The deciding factor is download size, not compute:

```
laptop A (recommender)   47 MB     minutes to fetch
laptop B (QA QDoRA)     534 MB     the 523 MB file that has dropped repeatedly
```

So **give Laptop B the better network connection**, or the machine already
holding `qa_train.jsonl`. If one laptop is on a flaky link, that one takes the
recommender.

## Setup - run on each laptop

```bash
git lfs install
git clone https://github.com/Hariprasath-5128/Medicine-Prescription-Engine.git
cd Medicine-Prescription-Engine
pip install -r requirements.txt

# Blackwell (sm_120) needs cu128. A cu121 wheel silently runs on CPU.
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then, per machine:

```bash
python scripts/setup_machine.py --role recommender   # laptop A
python scripts/setup_machine.py --role qa            # laptop B
```

This pulls **only that role's LFS objects**, checks the GPU, and refuses to
report READY if anything is a pointer file rather than real content. Cloning
everything on both machines would burn ~1.2 GB of a 1 GB/month LFS allowance.

## Run

**Laptop A** (~2-3 h):

```bash
python scripts/03_train_recommender.py --fp16 --budget-hours 3
```

**Laptop B** (the rest of the day):

```bash
python scripts/04_train_qa_dora.py --check          # verify cu128 FIRST
python scripts/04_train_qa_dora.py --budget-hours 20
```

Both stages measure real throughput on their own GPU before committing, and
subsample rather than overrun the budget. Two identical 5070s should report
near-identical numbers - a large discrepancy means one machine is thermally
throttled or on the wrong torch build.

## Merge

Only `artifacts/` needs to move, and only at the end. Copy Laptop A's
`artifacts/recommender/` (~1.3 GB) onto Laptop B, or vice versa - over a USB
drive or LAN, **not** through GitHub. Model weights do not belong in git; the
repo already ignores `artifacts/` and `*.bin`.

Then on whichever machine holds both:

```bash
python scripts/00_setup_evidence_db.py --verify   # 632 MB, needed for evaluation
python scripts/setup_machine.py --role eval
python scripts/05_evaluate.py --all
```

## Suggested timeline

| Time | Laptop A | Laptop B |
|---|---|---|
| 0:00 | clone + setup (47 MB) | clone + setup (534 MB) |
| 0:20 | `--check`, smoke test 2k rows | `--check`, smoke test `--limit 500` |
| 0:40 | **recommender starts** | **QA QDoRA starts** |
| ~3:30 | done; copy evidence DB | still running |
| ~4:00 | idle - see below | still running |
| ~8:00 | receives QA adapter | done |
| ~8:30 | `05_evaluate.py --all`, serve | idle |

**Do the smoke tests.** Ten minutes on each machine is far cheaper than
discovering a bad torch build eight hours into the QA run.

### Using Laptop A's idle time

Once the recommender finishes, that machine is free for ~4 h. Useful options,
best first:

1. **A second recommender with a different seed or encoder.** Train
   `--encoder base` alongside the large one and keep whichever scores better on
   val macro-F1. Cheap, and gives a real comparison rather than an assumption.
2. **Serve the API/UI** against the finished recommender so you can exercise the
   RAG path while the QA stage still runs. The system works without the QA model.
3. **Longer evaluation** - `05_evaluate.py --retrieval --n-queries 500` for a
   tighter confidence interval than the default 100.

## If one machine fails

The stages are independent, so a failure costs that stage only:

- **Recommender fails** -> the API and UI cannot recommend, but evidence search
  still works. Rerun on either laptop; it is only ~3 h.
- **QA fails** -> nothing else is affected. The recommender plus RAG is the
  functioning system; the QA fine-tune is the optional stage.

## What NOT to do

- **Do not split one model across both laptops.** No NVLink, no fast
  interconnect - gradient sync over consumer wifi is slower than one GPU alone.
- **Do not push `artifacts/` to git.** 1.3 GB of weights would blow the LFS
  quota that already holds 581 MB of data.
- **Do not run both stages on one laptop concurrently.** 6.71 GB + ~8 GB exceeds
  12 GB; they would fight for VRAM and both slow to a crawl.
