"""Measure real throughput, then size the run to fit a wall-clock budget.

Written because an estimate derived from the wrong GPU was badly wrong: the
4 GB dev card timed BioLinkBERT-large at 45 s/step, which extrapolated to ~51 h
per epoch on a 5070. That number was meaningless - the card was spilling to
shared system RAM, and the encoder alone timed identically with both heads
removed.

So: never extrapolate across GPUs. Time a handful of real steps on the machine
that will do the training, then decide how many rows fit the budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Throughput:
    samples_per_sec: float
    sec_per_step: float
    batch_size: int
    peak_vram_gb: float

    def hours_for(self, n_rows: int, epochs: int = 1) -> float:
        return n_rows * epochs / self.samples_per_sec / 3600

    def rows_within(self, hours: float, epochs: int = 1) -> int:
        return int(self.samples_per_sec * 3600 * hours / max(epochs, 1))


def measure(step_fn, batch_size: int, warmup: int = 3, iters: int = 8) -> Throughput:
    """Time `step_fn` after a warmup, returning throughput and peak VRAM.

    Warmup matters: the first steps pay CUDA context setup, cuDNN autotuning
    and lazy allocator growth, and would otherwise dominate a short benchmark.
    """
    import torch

    for _ in range(warmup):
        step_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    for _ in range(iters):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    sec_per_step = elapsed / iters
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return Throughput(
        samples_per_sec=batch_size / sec_per_step,
        sec_per_step=sec_per_step,
        batch_size=batch_size,
        peak_vram_gb=peak,
    )


def plan(tp: Throughput, available_rows: int, budget_hours: float,
         epochs: int, label: str) -> tuple[int, str]:
    """Decide how many rows to train on, and explain the decision.

    Returns (rows_to_use, human-readable report).
    """
    full_hours = tp.hours_for(available_rows, epochs)
    lines = [
        f"--- {label} throughput (measured on this GPU) ---",
        f"  {tp.sec_per_step:.3f} s/step at batch {tp.batch_size} "
        f"= {tp.samples_per_sec:.1f} samples/s",
        f"  peak VRAM {tp.peak_vram_gb:.2f} GB",
        f"  full dataset: {available_rows:,} rows x {epochs} epochs "
        f"= {full_hours:.1f} h",
        f"  budget: {budget_hours:.1f} h",
    ]

    if full_hours <= budget_hours:
        lines.append(f"  -> FITS. Training on all {available_rows:,} rows.")
        return available_rows, "\n".join(lines)

    # Keep 10% slack: evaluation, checkpointing and the odd slow batch are not
    # in the measured step time.
    FLOOR = 1000  # below this a fine-tune is not worth running at all
    rows = int(tp.rows_within(budget_hours * 0.9, epochs))
    rows = max(FLOOR, min(rows, available_rows))
    projected = tp.hours_for(rows, epochs)

    lines += [
        f"  -> OVER BUDGET by {full_hours - budget_hours:.1f} h.",
        f"  -> Subsampling to {rows:,} rows (~{projected:.1f} h).",
    ]
    # Say so plainly when even the floor overruns, rather than reporting a
    # projection that quietly exceeds the budget the caller asked for.
    if projected > budget_hours:
        lines += [
            f"  !! Even the {FLOOR:,}-row floor needs {projected:.1f} h, over the "
            f"{budget_hours:.1f} h budget.",
            "  !! This GPU is too slow for this model. Use a smaller base model "
            "(--small), or raise --budget-hours.",
        ]
    return rows, "\n".join(lines)
