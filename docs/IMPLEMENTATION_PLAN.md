# Implementation Plan

## 1. The finding that reshaped this plan

The intended training set — Kaggle `abdelrahmangamal236/medical-conditions50000` —
advertises 50,000 rows. **It contains 20.**

```
rows=50,000  cols=5
column                       distinct     verdict
ID                             50,000
Symptoms/Question                  20     TEMPLATED (~2,500 copies each)
Disease Prediction                 20     TEMPLATED (~2,500 copies each)
Recommended Medicines              20     TEMPLATED (~2,500 copies each)
Advice                             20     TEMPLATED (~2,500 copies each)

distinct content rows : 20 (0.0%)
duplicate rows        : 49,980 (100.0%)
```

Reproduce it yourself:

```bash
python scripts/01_audit_dataset.py "data/raw/medical_question_answer_dataset_50000.csv" \
    --text-col "Symptoms/Question" --label-col "Disease Prediction"
```

Why this rules the dataset out for training:

- **Any train/test split leaks.** With 20 distinct rows copied 2,500 times each,
  every test row is also a training row. Accuracy would read ~100% and mean
  nothing. A reviewer checks `df.duplicated().sum()` first and the result is
  indefensible.
- **A `dict` reproduces it exactly.** Twenty fixed input strings map to twenty
  fixed outputs. No model can beat a lookup table here, and none can generalise
  to a twenty-first symptom.
- **There is no language to learn.** Each "symptom" is one frozen 2–5 word
  phrase, so an NLP model sees twenty tokens-sequences, not natural text.

The dataset is kept in `data/raw/` and used as a **demo lookup table and a
negative control** — a worked example of the audit catching a bad corpus. It is
not trained on.

## 2. What replaced it

| Purpose | Dataset | Real scale (verified) |
|---|---|---|
| Drug recommender | UCI Drug Reviews (`jessicali9530/kuc-hackathon-winter-2018`) | **95,456 deduplicated rows**, 90 conditions, 2,044 drugs |
| QA fine-tune | **UltraMedical** (`TsinghuaC3I/UltraMedical`) | 409,593 rows → **181,700 pairs** kept |
| Evidence / citations | SERA Chroma index (bge-m3) | 56,407 chunks with `source_url` |

Measured cleaning trail for the drug reviews:

```
loaded 161,297 rows
after cleaning       : 157,823
after dedup          : 110,236   (removed 47,587 exact copies)
after rare-class cut :  95,456   (90 conditions)
split train/val/test : 81,137 / 7,159 / 7,160
leakage check passed: no review text shared across splits
```

Those 47,587 duplicates matter: leaving them in would have re-created the exact
flaw that disqualified the 50k file.

### Why UltraMedical, not the SERA chunks

The first design synthesised QA pairs from the SERA index by templating section
names into questions. Two measurements killed it:

- **Yield was 20.5%.** The corpus has **9,393 distinct section names**, not a
  clean enum, so 8 templates matched only 11,560 of 56,407 chunks. The discarded
  79% included the most drug-relevant text in the corpus ("WHY IS THIS MEDICATION
  PRESCRIBED?", 1,068 chunks).
- **It was self-distillation.** Training on text the RAG index already holds adds
  no knowledge, teaches only format, and encourages the model to recite corpus
  text from memory *uncited* - defeating the citation layer.

UltraMedical (NeurIPS 2024 D&B Spotlight) is the dataset the base model was
itself trained on. Audited with the same script that rejected the 50k file:

```
409,593 rows | 96.07% distinct questions | question words 1/77/1390
composition: Exam 211,707 | Open-End 109,198 | Literature 88,688
```

Exam rows are dropped: their answers are bare option letters ("D"), which teach
terseness rather than explanation. Keeping the two prose types and dropping ~30
degenerate rows yields **181,700 pairs**:

```
PubMedQA 88,685 | Medical-Instruct-120k 25,780 | WikiInstruct 23,283
MedInstruct-52k 22,959 | ChatDoctor 15,126 | MedQuad 5,867
split train/val = 178,066/3,634 (no shared questions)
```

Roles are now cleanly separated: **UltraMedical supplies instruction behaviour,
SERA supplies verifiable evidence.**

## 3. Hardware reality

`nvidia-smi` on this machine reports an **RTX 2050, 4.3 GB, sm_86** — not a 5070.
The plan therefore splits by machine:

| Stage | Machine | Measured / budgeted VRAM |
|---|---|---|
| Recommender, BioLinkBERT-large batch 8 | 5070 | **6.71 GB measured** |
| Recommender, BioLinkBERT-large batch 4 | either card | **3.01 GB measured** |
| Recommender, base encoders | either card | **2.6 GB measured** |
| Evidence retrieval (bge-m3) | either, CPU by default | ~2.2 GB if moved to GPU |
| QA DoRA fine-tune, 8B | 5070 (12 GB) only | ~7.9 GB budgeted |
| QA DoRA fine-tune, Phi-3-mini | 5070, or 4 GB at seq 512 | ~4.5 GB |

8B QDoRA budget at seq 1024, batch 1 × accumulation 16:

```
4-bit NF4 base weights            ~4.7 GB
DoRA adapters + grads (bf16)      ~0.5 GB
paged AdamW-8bit states           ~0.2 GB
activations (grad checkpointing)  ~2.5 GB
                                  -------
                                  ~7.9 GB   under 12 GB with headroom
```

Published 8B QLoRA runs at **seq 2048 / batch 4 peak at 14–15 GB and OOM a 12 GB
card**, which is why `max_seq_length` is pinned to 1024.

### Throughput: do not benchmark on the 2050

Measured on the RTX 2050 (45W laptop-class, 4 GB):

```
BioLinkBERT-large  batch 4  45.26 s/step
BioLinkBERT-base   batch 8  11.26 s/step
encoder alone      batch 8  11.44 s/step   <- same as the full model
```

The third line is the important one: stripping both heads changed nothing, so
the two-head design is **not** the bottleneck. At 3,585 MiB of 4,096 MiB used
(87.5%), the Windows driver spills to shared system RAM and every step pays a
PCIe round-trip.

These numbers are therefore an artefact of this GPU and must not be
extrapolated to the 5070 - a desktop Blackwell card with 12 GB has no spill and
far more memory bandwidth. **Time one epoch on the 5070 before planning a full
run**; do not trust an estimate derived from the 2050.

### Blackwell caveat

The 5070 is `sm_120`. It needs a **cu128** PyTorch build; a cu121 wheel fails or
silently drops to CPU. `scripts/04_train_qa_dora.py --check` detects this before
a multi-hour run starts.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Windows download caveat (hit during this build)

The HuggingFace **Xet CDN backend fails silently** on this setup — the process
exits with no traceback and no output. Both training scripts now set
`HF_HUB_DISABLE_XET=1` by default. This cost an hour to diagnose; it is baked in
so you never hit it.

## 3b. Fitting one day on the 5070

Training runs on the RTX 5070 (12 GB VRAM, 24 GB system RAM) with a **one-day**
wall-clock budget. Two mechanisms keep that promise:

**Measured, never estimated.** `src/utils/budget.py` times real optimiser steps
on the actual GPU after a warmup, then sizes the dataset to the budget. This
exists because an estimate extrapolated from the 4 GB dev card was wrong by
~5x - that card was spilling to shared system RAM, and the encoder alone timed
identically with both heads removed.

**Stage ordering by cost and necessity.** `scripts/run_all.py` runs cheap,
required work first and gives the optional 8B fine-tune whatever time is left,
holding back 0.4 h so evaluation always runs.

Reference anchor for planning only (an RTX 3090 does 8B LoRA over 52k samples x
3 epochs in 78 min; QDoRA is ~15% slower). That suggests the full 178k x 2
epochs is feasible in well under a day on a 5070 - but the code measures rather
than relying on it.

### Accuracy settings enabled by 12 GB

| Setting | Was | Now | Why |
|---|---|---|---|
| DoRA rank | 16 | **32** (alpha 64) | ~+1% on medical benchmarks for ~0.3 GB |
| NEFTune | off | **alpha 5** | Embedding noise; free, improves instruction-following |
| Recommender batch | 8 | **16** | 6.71 GB measured at batch 8, so 16 fits |
| Early stopping | none | **patience 2** | Stops when val macro-F1 plateaus |

## 4. Model choices

| Stage | Model | Why |
|---|---|---|
| QA fine-tune | `TsinghuaC3I/Llama-3-8B-UltraMedical` | Beats **Meditron-70B** on MedQA/MedMCQA/PubMedQA. Meditron-7B is Llama-2-era; independent 2026 evaluation ranked Meditron-70B *last* among 70B models on a neurology board exam. |
| QA fallback | `microsoft/Phi-3-mini-4k-instruct` | Trains comfortably in 12 GB with room to spare. |
| Recommender | `michiyasunaga/BioLinkBERT-large` | **SOTA on BLURB and MedQA-USMLE**; 335.7M params. Pretrained on PubMed *with citation links*, so it models relations between documents rather than isolated abstracts. |
| Recommender fallback | `michiyasunaga/BioLinkBERT-base` / `BiomedBERT-base` | 110M, ~2.6 GB - for fast iteration on the 4 GB card. |
| Embeddings | `BAAI/bge-m3` | **Forced choice** — must match what SERA used to build the index. |

DoRA (`use_dora=True`) needs `peft >= 0.10`; installed here is 0.14.0.

## 5. Architecture

```
user text
   │
   ├─► safety triage ──────────► red-flag terms short-circuit to "seek care now"
   │
   ├─► recommender (PubMedBERT, two heads over one encoder)
   │      ├─ condition head : softmax over 90 conditions
   │      └─ drug head      : sigmoid over 2,044 drugs  (multi-label)
   │
   ├─► retriever (bge-m3 → Chroma, cosine distance ≤ 0.55)
   │      └─ returns chunk + section + relevance + source_url, or nothing
   │
   └─► Recommendation
          condition + ranked drugs + numbered citations
          or an explicit UNVERIFIED marker when nothing clears threshold
```

The drug head is **multi-label on purpose**. A condition legitimately has several
valid drugs, so forcing one "correct" answer would be clinically wrong and would
inflate the metric. It is supervised with every drug attested for that condition
in the training split — a review of Sertraline for Depression does not make
Fluoxetine a negative example.

The recommender and retriever are **independent**. Retrieval confirms or fails to
confirm what the recommender proposed; it never supplies the recommendation.

## 6. Measured results so far

Retrieval, 60 self-supervised probes against the live index:

```
mrr        : 0.4881
recall@1   : 0.4667
recall@3   : 0.5000
recall@5   : 0.5333
```

Note the first version of this probe returned `recall@1 == recall@3 == recall@5`
— a flat curve that is a red flag, not a result. Two corpus quirks caused it:

1. `section` is **free text, not an enum**. Some sections are whole disease names
   (`"11q partial monosomy syndrome"`), so concatenating focus + section produced
   queries about two unrelated diseases.
2. `document_id` is **too strict** as a relevance label — the corpus holds
   near-duplicate documents on the same topic (`0000579` and `0000580` are both
   "Human Papillomavirus (HPV) Vaccine"), so returning the sibling scored as a
   miss.

The probe now filters to heading-like sections and matches on normalised
`question_focus`. Recall rises with K as it should.

## 7. Honest limitations

- **The corpus is disease-information text** (MedQuAD / Genetics Home Reference),
  not a drug-trial corpus and not PubMed abstracts. It can confirm a *condition*
  is described as stated; it is **not** drug-efficacy evidence. Do not label this
  "PubMed RAG" in a write-up — the sources are `ghr.nlm.nih.gov` and friends.
- **Grounding rate is 100% on the sampled conditions (50/50), measured.** I
  expected this to be poor; it is not, because the corpus includes MedlinePlus
  alongside the genetics material. The caveat that survives is narrower: a
  retrieved chunk confirms the *condition* is described as stated, and is not
  evidence that the recommended drug is the right choice.
- **Association is not prescription.** The drug head learns what patients wrote
  reviews about. Popular drugs are over-represented; the output carries no dose,
  route, frequency, duration, contraindication or interaction check.
- **This is decision support, not a prescriber.** Every response carries the
  disclaimer and requires clinician review.

## 8. Naming

Call stage 2 a **Prescription Recommendation Engine**, not a generation model.
The data supports `condition → commonly associated drugs`; it does not support
`patient → dose/route/frequency/duration`.

## 9. Build order

| # | Step | Command | Needs |
|---|---|---|---|
| 0 | Copy evidence index | `python scripts/00_setup_evidence_db.py --verify` | SERA present |
| 1 | Audit any dataset | `python scripts/01_audit_dataset.py <csv>` | — |
| 2 | Prepare data | `python scripts/02_prepare_data.py --all` | kaggle.json |
| 3 | Train recommender | `python scripts/03_train_recommender.py` | 4 GB card is fine |
| 4 | DoRA QA fine-tune | `python scripts/04_train_qa_dora.py --check` first | **5070** |
| 5 | Evaluate | `python scripts/05_evaluate.py --all` | — |
| 6 | Serve | `uvicorn src.api.main:app --port 8000` | step 3 |
| 7 | UI | `streamlit run src/ui/app.py` | step 3 |

Steps 0–3, 5, 6 and 7 run on the 4 GB machine. Only step 4 needs the 5070.
