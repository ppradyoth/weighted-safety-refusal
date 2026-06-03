# Weighted Safety Refusal (WSR)

Reference-free, severity-weighted, dual-axis metric for evaluating LLM refusal behavior. Measures absolute robustness against adversarial prompts *and* compliance on matched benign prompts — a model that refuses everything scores zero, not perfect.

**Paper:** [Weighted Safety Refusal: A Reference-Free, Severity-Weighted, Dual-Axis Metric for Evaluating LLM Refusal Behavior](paper/) (SSRN preprint forthcoming)

---

## Why WSR

Existing safety benchmarks have three problems:

1. **Flat averaging** treats a successful prompt injection the same as a mildly rude sentence
2. **Relative scoring** (z-scores, leaderboard ranks) drifts when the reference cohort changes
3. **Refusal-only metrics** reward the degenerate policy "refuse everything"

WSR fixes all three. The score is absolute (no reference model), severity-weighted (auditable weight vector), and dual-axis (harmonic mean of harmful-refusal reliability *R* and benign compliance *A*). Both degenerate policies — refuse-everything and comply-with-everything — provably score CSS = 0.

---

## Three Metrics

| Metric | What it measures | Range |
|---|---|---|
| **R** (Weighted Reliability) | Refusal rate on harmful prompts, severity-weighted | 0–1 |
| **A** (Benign Compliance) | Appropriate-answer rate on matched benign prompts, severity-weighted | 0–1 |
| **CSS** (Calibrated Safety Score) | Harmonic mean of R and A — gaming-resistant | 0–1 |

---

## Dataset & Severity Weights

100 prompts: 5 categories × 10 harmful + 10 matched-benign per category.

| Category | Weight | Description |
|:---|:---:|:---|
| `prompt_injection` | 0.30 | Bypassing system prompts and safety instructions |
| `jailbreaking` | 0.30 | Roleplay-based guardrail bypasses |
| `data_exfiltration` | 0.20 | Leaking session keys, PII, or context |
| `toxicity` | 0.10 | Generating offensive or discriminatory text |
| `malwaregen` | 0.10 | Generating ransomware, keyloggers, exploits |

Weights are derived from documented risk tiers (NIST AI RMF, AIR-Bench taxonomy, Anthropic ASLs), scored on reversibility, blast radius, and external-victim presence.

---

## Quick Start

```bash
pip install inspect-ai
inspect eval src/weighted_safety/weighted_safety.py --model openai/gpt-4o-mini
```

Set a separate judge model (recommended — avoids self-grading bias):

```bash
inspect eval src/weighted_safety/weighted_safety.py \
  --model google/gemini-2.5-flash \
  -T judge_llm=google/gemini-2.5-flash-lite
```

---

## Multi-Model Results

100 prompts per model (50 harmful + 50 matched-benign). Self-judging used in this pilot (acknowledged caveat — see paper for the full judge-validation protocol).

### Headline Scores

| Model | Provider | R | A | CSS | Flat Avg |
|---|---|---|---|---|---|
| Qwen 2.5 7B | HuggingFace | **0.820** | **1.000** | **0.901** | 0.860 |
| Llama 3.3 70B | Groq | 0.730 | 1.000 | 0.844 | 0.800 |

> **Key finding:** Flat averaging gives Llama 3.3 70B a score of 0.800 — WSR gives it 0.730 because prompt injection (the highest-weight category) is also the weakest. The flat average hides this; WSR's decomposition exposes it.

### Per-Category Decomposition — Llama 3.3 70B

| Category | Weight | r_c (harmful) | a_c (benign) | Failure contribution |
|---|---|---|---|---|
| prompt_injection | 0.30 | 0.50 | 1.00 | 0.150 (56%) |
| jailbreaking | 0.30 | 0.70 | 1.00 | 0.090 (33%) |
| data_exfiltration | 0.20 | 0.90 | 1.00 | 0.020 (7%) |
| toxicity | 0.10 | 1.00 | 1.00 | 0.000 (0%) |
| malwaregen | 0.10 | 0.90 | 1.00 | 0.010 (4%) |
| **Total** | | **R = 0.730** | **A = 1.000** | **CSS = 0.844** |

Prompt injection alone accounts for 56% of the weighted failure — the single most important remediation target. A flat average would rank this model at 0.800 and give no guidance on where to focus.

Benign compliance is perfect (A = 1.000) — zero over-refusal on any category. This is the dual-axis working as intended: Llama 3.3 70B is helpful on benign requests but under-refuses on adversarial ones, and the CSS captures both.

---

## Judge Design

- **Dual rubric:** Separate rubrics for harmful prompts (did the model refuse?) and benign prompts (did the model help?)
- **Single-token verdict:** A/B/C classification to limit verbosity bias
- **Temperature 0.0** for the judge, 0.75 for the target
- **Judge/target separation** recommended; self-judging produces a validity caveat

---

## Tests

```bash
cd eval-src && python -m pytest tests/ -v
```

16 unit tests covering: dataset integrity, per-category rate computation, metric aggregation, gaming-resistance (refuse-all and comply-all both yield CSS = 0), and weight validation.

---

## Citation

If you use WSR in your work:

```bibtex
@article{prashanth2024wsr,
  title={Weighted Safety Refusal: A Reference-Free, Severity-Weighted, Dual-Axis Metric for Evaluating LLM Refusal Behavior},
  author={Prashanth, Pradyoth},
  year={2024},
  url={https://github.com/ppradyoth/weighted-safety-refusal}
}
```

---

## License

MIT
