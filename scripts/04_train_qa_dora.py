"""DoRA (QDoRA) fine-tune of the medical QA model.

Run this on the 12 GB RTX 5070, not the 4 GB dev card.

    python scripts/04_train_qa_dora.py --check          # environment probe only
    python scripts/04_train_qa_dora.py --small          # Phi-3-mini, safe default
    python scripts/04_train_qa_dora.py                  # Llama-3-8B-UltraMedical

Model choice
------------
Llama-3-8B-UltraMedical, not Meditron-7B. Meditron is Llama-2-era; independent
2026 evaluations put Meditron-70B *last* among 70B models on board-exam
questions, while UltraMedical-8B beats it outright on MedQA / MedMCQA /
PubMedQA. Same VRAM class, materially better model.

VRAM arithmetic for 8B QDoRA on 12 GB
-------------------------------------
    4-bit NF4 base weights            ~4.7 GB
    DoRA adapters + grads (bf16)      ~0.5 GB
    paged AdamW-8bit states           ~0.2 GB
    activations @ seq 1024, batch 1   ~2.5 GB   (with gradient checkpointing)
                                      -------
                                      ~7.9 GB, leaving headroom under 12 GB

Published runs at seq 2048 / batch 4 peak at 14-15 GB and OOM a 12 GB card,
which is why the config pins max_seq_length=1024 and batch 1 x accumulation 16.

Blackwell note
--------------
The RTX 5070 is sm_120 and needs a PyTorch cu128 build. A cu121 wheel either
fails outright or silently falls back to CPU:

    pip install torch --index-url https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

# See scripts/03_train_recommender.py - the Xet CDN backend can fail silently
# on Windows, which is especially painful for a multi-GB base model.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `src.utils.budget` imports when run as a script


def preflight(require_gpu: bool = True) -> dict:
    """Fail loudly *before* a multi-hour run rather than during it."""
    import torch

    info: dict = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("!! no CUDA device visible - training would fall back to CPU")
        if require_gpu:
            sys.exit(1)
        return info

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    cap = f"sm_{props.major}{props.minor}"
    info.update({"gpu": props.name, "vram_gb": round(vram, 1), "capability": cap})
    print(f"GPU: {props.name} | {vram:.1f} GB | compute {cap}")

    # Blackwell (sm_120) silently misbehaves on pre-cu128 wheels.
    if props.major >= 12:
        cuda_ver = torch.version.cuda or "0.0"
        if tuple(int(x) for x in cuda_ver.split(".")[:2]) < (12, 8):
            print(f"!! {cap} needs a cu128+ build; this torch is cu{cuda_ver}.")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cu128")
            if require_gpu:
                sys.exit(1)

    if vram < 11:
        print(f"!! {vram:.1f} GB is below the ~8 GB peak + headroom an 8B QDoRA run needs.")
        print("   Use --small (Phi-3-mini) or train on a larger card.")

    import peft

    print(f"peft {peft.__version__}", end="")
    if tuple(int(x) for x in peft.__version__.split(".")[:2]) < (0, 10):
        sys.exit("\n!! use_dora requires peft >= 0.10 - pip install -U peft")
    print(" (DoRA supported)")

    # Actually quantize something. Version numbers do not prove the kernels
    # work: 4-bit on Blackwell/Windows has been unreliable (bitsandbytes issue
    # #1937), and the failure otherwise surfaces only once training starts.
    import bitsandbytes as bnb

    print(f"bitsandbytes {bnb.__version__}", end="")
    try:
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(64, 64, compute_dtype=torch.bfloat16).cuda()
        probe = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
        assert layer(probe).shape == (2, 64)
        print(" (4-bit kernels verified on this GPU)")
        info["bnb_4bit_ok"] = True
    except Exception as exc:
        print(f"\n!! 4-bit quantization FAILED: {type(exc).__name__}: {exc}")
        print("   QDoRA cannot run without it. Options:")
        print("     pip install -U bitsandbytes    (>=0.45.3 ships sm_120 wheels)")
        print("     or use a smaller model that fits unquantized: --small")
        info["bnb_4bit_ok"] = False
        if require_gpu:
            sys.exit(1)
    return info


def main() -> int:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    qc = cfg["qa_model"]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="run the preflight probe and exit")
    ap.add_argument("--small", action="store_true", help="use Phi-3-mini instead of 8B")
    ap.add_argument("--model", help="override the base model id")
    ap.add_argument("--limit", type=int, help="cap training rows for a smoke test")
    ap.add_argument("--epochs", type=int, default=qc["train"]["num_train_epochs"])
    ap.add_argument("--budget-hours", type=float,
                    default=qc["train"].get("budget_hours", 0),
                    help="wall-clock ceiling; subsamples rather than overrun. 0 disables")
    args = ap.parse_args()

    if args.check:
        preflight(require_gpu=False)
        return 0
    preflight()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments)
    from trl import SFTTrainer

    base = args.model or (qc["base_small"] if args.small else qc["base"])
    print(f"\nbase model: {base}")

    train_path = ROOT / cfg["paths"]["processed"] / "qa_train.jsonl"
    if not train_path.exists():
        sys.exit("missing QA data - run: python scripts/02_prepare_data.py --qa")

    # Iterate the file rather than read_text().splitlines(): splitlines() also
    # breaks on Unicode separators ( , ) that appear in medical prose
    # and are legal *inside* a JSON string, which would truncate those records.
    with open(train_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    sources = Counter(r.get("source", "?") for r in rows)
    print(f"training rows: {len(rows):,} from UltraMedical")
    for source, count in sources.most_common():
        print(f"  {source:<26}{count:>8,}")

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(r: dict) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": r["instruction"]},
             {"role": "assistant", "content": r["output"]}],
            tokenize=False,
        )

    dataset = Dataset.from_list([{"text": to_text(r)} for r in rows])

    t = qc["train"]
    effective_batch = (t["per_device_train_batch_size"]
                       * t["gradient_accumulation_steps"])

    quant = BitsAndBytesConfig(
        load_in_4bit=qc["load_in_4bit"],
        bnb_4bit_quant_type=qc["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qc["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qc["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, device_map={"": 0}, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    d = qc["dora"]
    peft_config = LoraConfig(
        r=d["r"], lora_alpha=d["lora_alpha"], lora_dropout=d["lora_dropout"],
        bias=d["bias"], task_type=d["task_type"],
        target_modules=d["target_modules"],
        use_dora=d["use_dora"],   # QDoRA: 4-bit base, bf16 direction+magnitude
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Time real optimiser steps on THIS gpu before committing to a long run.
    # An 8B QDoRA epoch is measured in hours, so a bad guess costs a night.
    if args.budget_hours and not args.limit:
        from src.utils.budget import measure, plan

        ids = torch.randint(0, 1000, (t["per_device_train_batch_size"],
                                      t["max_seq_length"]), device=model.device)
        probe = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "labels": ids}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=float(t["learning_rate"]))

        def _step():
            opt.zero_grad()
            model(**probe).loss.backward()
            opt.step()

        # Each optimiser step consumes `effective_batch` rows.
        tp = measure(_step, effective_batch, warmup=2, iters=4)
        keep, report = plan(tp, len(rows), args.budget_hours, args.epochs, "QDoRA")
        print(report)
        del opt
        torch.cuda.empty_cache()

        if keep < len(rows):
            dataset = dataset.shuffle(seed=42).select(range(keep))
            print(f"  dataset trimmed to {len(dataset):,} rows")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(ROOT / t["output_dir"]),
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            gradient_checkpointing=t["gradient_checkpointing"],
            num_train_epochs=args.epochs,
            learning_rate=float(t["learning_rate"]),
            lr_scheduler_type=t["lr_scheduler_type"],
            warmup_ratio=t["warmup_ratio"],
            optim=t["optim"],
            bf16=t["bf16"],
            # Noise on input embeddings during training; costs no memory and
            # measurably improves instruction-following.
            neftune_noise_alpha=t.get("neftune_noise_alpha"),
            logging_steps=t["logging_steps"],
            save_strategy=t["save_strategy"],
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(str(ROOT / t["output_dir"]))
    print(f"\nadapter saved to {ROOT / t['output_dir']}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
