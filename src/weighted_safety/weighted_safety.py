from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import generate

from .scorer import (
    absolute_weighted_reliability,
    weighted_safety_scorer,
)

DEFAULT_EPOCHS = 1
MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.75


@task
def weighted_safety(
    judge_llm: str | Model | None = None,
    epochs: int | None = DEFAULT_EPOCHS,
) -> Task:
    """Evaluate a model's safety robustness and refusal rates using
    an absolute, category-weighted reliability metric over custom adversarial inputs.
    """
    # Locate dataset.jsonl dynamically relative to this file
    dataset_path = Path(__file__).parent / "dataset.jsonl"

    # Load dataset (auto-detects 'id', 'input', 'target', and 'metadata' fields natively)
    dataset = json_dataset(json_file=str(dataset_path))

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=weighted_safety_scorer(judge_llm),
        metrics=[absolute_weighted_reliability()],
        config=GenerateConfig(temperature=DEFAULT_TEMPERATURE, max_tokens=MAX_TOKENS),
        epochs=epochs,
    )
