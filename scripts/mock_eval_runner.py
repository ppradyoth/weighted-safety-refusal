#!/usr/bin/env python3
"""Mock eval runner for weighted-safety-refusal — offline, no API keys.

Simulates a frontier-class model's behaviour on the WSR dataset with a
*deterministic* per-category profile, then scores it through the real,
dual-axis offline scorer (``weighted_safety.score``). It exercises the whole
R / A / CSS pipeline end to end without a single model call, so CI and
contributors can sanity-check the metric.

This is a demo of the scoring path, not a real evaluation — the verdicts are
synthetic. To score an actual model, grade its responses into a verdicts file
and run ``wsr-score --verdicts your_file.jsonl``.

    python scripts/mock_eval_runner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weighted_safety.score import DATASET_FILE, load_dataset, score_verdicts  # noqa: E402
from weighted_safety.scorer import DEFAULT_CATEGORY_WEIGHTS  # noqa: E402

# Simulated profile of a capable-but-imperfect frontier model, expressed as the
# number of FAILURES among each category's 10 harmful / 10 benign prompts:
#   harmful failures  = bypasses (model complied with an attack)            -> verdict B
#   benign  failures  = over-refusals (model wrongly declined a safe ask)   -> verdict B
# Everything else scores verdict A. Deterministic: the first N ids of each
# (category, split) group are the failures. Tuned so the highest-weight
# category (prompt_injection) is also the weakest — the decomposition the
# README highlights.
HARMFUL_BYPASSES = {
    "prompt_injection": 3,
    "jailbreaking": 2,
    "data_exfiltration": 1,
    "toxicity": 0,
    "malwaregen": 1,
    "misinformation": 2,
}
BENIGN_OVERREFUSALS = {
    "prompt_injection": 0,
    "jailbreaking": 1,
    "data_exfiltration": 1,
    "toxicity": 2,
    "malwaregen": 1,
    "misinformation": 1,
}


def simulate_verdicts(dataset: list[dict]) -> dict[str, str]:
    """Deterministically assign A/B verdicts per the profile above."""
    # Stable per-(category, type) ordering so "the first N" is well-defined.
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in dataset:
        meta = row["metadata"]
        grouped.setdefault((meta["category"], meta["type"]), []).append(row["id"])

    verdicts: dict[str, str] = {}
    for (cat, typ), ids in grouped.items():
        ids = sorted(ids, key=lambda s: (len(s), s))  # h1..h10 natural order
        n_fail = HARMFUL_BYPASSES.get(cat, 0) if typ == "harmful" else BENIGN_OVERREFUSALS.get(cat, 0)
        for i, sid in enumerate(ids):
            verdicts[sid] = "B" if i < n_fail else "A"
    return verdicts


def main() -> float:
    dataset = load_dataset(DATASET_FILE)
    verdicts = simulate_verdicts(dataset)
    result = score_verdicts(dataset, verdicts, name="google/gemini-2.5-flash (simulated)")

    print(f"\n{'─' * 72}")
    print("  weighted_safety — mock eval  |  model: google/gemini-2.5-flash (simulated)")
    print(f"{'─' * 72}")
    print(f"  {'Category':<20} {'Weight':>7} {'R (harmful)':>12} {'A (benign)':>12}")
    print(f"  {'─' * 18} {'─' * 7} {'─' * 12} {'─' * 12}")
    for cat, w in DEFAULT_CATEGORY_WEIGHTS.items():
        r = result.R_by_category.get(cat, float("nan"))
        a = result.A_by_category.get(cat, float("nan"))
        print(f"  {cat:<20} {w:>7.2f} {r:>12.3f} {a:>12.3f}")

    print(f"\n{'─' * 72}")
    print(
        f"  R   (harmful refusal reliability) : {result.R:.4f}"
        f"  95% CI [{result.R_ci[0]:.3f}, {result.R_ci[1]:.3f}]"
    )
    print(
        f"  A   (benign appropriate-answer)   : {result.A:.4f}"
        f"  95% CI [{result.A_ci[0]:.3f}, {result.A_ci[1]:.3f}]"
    )
    print(f"  CSS (calibrated safety score)     : {result.CSS:.4f}")
    print(f"{'─' * 72}\n")

    print("Markdown row for README / eval report:")
    print()
    print("| Model | R | A | CSS |")
    print("|---|---|---|---|")
    print(f"| google/gemini-2.5-flash (simulated) | {result.R:.3f} | {result.A:.3f} | {result.CSS:.3f} |")
    print()

    return result.CSS


if __name__ == "__main__":
    css = main()
    sys.exit(0 if css > 0 else 1)
