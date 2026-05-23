# Weighted Safety Refusal: Absolute Safety Robustness Evaluations

The **Weighted Safety Refusal** evaluation suite is a safety benchmark designed to evaluate Large Language Model robustness against diverse adversarial attacks (prompt injection, jailbreaking, data exfiltration, toxicity, malware generation).

Rather than relying on comparative, relative benchmarks (like Z-scores, which are calculated against shifting, potentially uncalibrated reference model set distributions), this evaluation measures **absolute, severity-weighted category failure rates** to yield a transparent reliability metric (0.0 to 1.0) specific to your model.

---

## 🧠 Design Philosophy

Traditional safety benchmarks often use standard statistical z-scoring relative to a periodic baseline set of LLMs. This has significant limitations:
1. **Calibration Dependency**: Relative scores are vulnerable to average baseline shifts.
2. **Obscured Failures**: A high relative z-score can mask absolute safety failures if the reference models perform even worse.
3. **No Risk Distinction**: Direct prompt injections, roleplays, and data exfiltration are treated with equal severity.

To address these limitations, this suite:
* Measures **absolute performance** directly on the provided dataset.
* Assigns **risk-adjusted severity weights** to each vulnerability category.
* Integrates a custom async model-graded grader using professional safety-auditing rubrics.

---

## 📊 Dataset & Severity Weights

The dataset (`dataset.jsonl`) contains custom adversarial prompts across five key categories, weighted according to real-world risk severity:

| Category | Severity Weight | Description |
| :--- | :---: | :--- |
| **`prompt_injection`** | **0.30** | Bypassing core system prompts and safety instructions. |
| **`jailbreaking`** | **0.30** | Roleplay-based guardrail bypasses (e.g., Developer Mode exploits). |
| **`data_exfiltration`** | **0.20** | Attempting to leak pre-loaded session keys or sensitive PII. |
| **`toxicity`** | **0.10** | Propensity to generate offensive or discriminatory text. |
| **`malwaregen`** | **0.10** | Generating cryptographic ransomware or malicious scripts. |

---

## ⚡ Quick Start

To run this evaluation using `inspect_ai` on any target model (e.g. `gpt-4o-mini`), run:

```bash
inspect eval src/weighted_safety/weighted_safety.py --model openai/gpt-4o-mini
```

### Grader Configuration
By default, the task uses whatever model is configured for the `grader` role. You can explicitly set a judge model via:

```bash
inspect eval src/weighted_safety/weighted_safety.py -T judge_llm=openai/gpt-4o-mini
```
