"""
Math dataset loader for SFT training.

Fixes the original bug: OpenR1-Math-220k examples were trained on the bare
`answer` field (e.g. "364") instead of the worked `solution`.

Supported datasets (chosen automatically from the dataset name):
  - open-r1/OpenR1-Math-220k   -> problem / solution   (olympiad level, hard for 0.5B)
  - meta-math/MetaMathQA       -> query / response     (step-by-step, recommended)
  - openai/gsm8k               -> question / answer    (calculator tags stripped)

Function names and signatures are unchanged, so train_sft_expert.py needs no edits.
"""

import random
import re
from typing import Dict, List, Optional

from datasets import load_dataset

_GSM8K_CALC_TAG = re.compile(r"<<.*?>>")


def _load_raw(dataset_name: str, split: str):
    if "gsm8k" in dataset_name.lower():
        return load_dataset(dataset_name, "main", split=split)
    return load_dataset(dataset_name, split=split)


def _format_example(example: dict, dataset_name: str) -> Optional[Dict[str, str]]:
    """Map one raw example to {instruction, input, output}, or None if unusable."""
    name = dataset_name.lower()

    if "gsm8k" in name:
        question = example.get("question")
        answer = example.get("answer") or ""
        answer = _GSM8K_CALC_TAG.sub("", answer)
        if "####" in answer:
            reasoning, final = answer.split("####", 1)
            answer = f"{reasoning.strip()}\nThe answer is: {final.strip()}"
    elif "metamath" in name:
        question = example.get("query")
        answer = example.get("response")
    else:  # OpenR1-Math-220k (and similar): use the worked solution, NOT `answer`
        question = example.get("problem")
        answer = example.get("solution")

    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return None
    return {"instruction": question, "input": "", "output": answer}


def _load_formatted(dataset_name: str, split: str, subset_size: int,
                    seed: int) -> List[Dict[str, str]]:
    """Shuffle deterministically and collect up to subset_size VALID examples."""
    dataset = _load_raw(dataset_name, split)

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    examples: List[Dict[str, str]] = []
    for i in indices:
        formatted = _format_example(dataset[i], dataset_name)
        if formatted is not None:
            examples.append(formatted)
            if len(examples) >= subset_size:
                break
    return examples


def load_math_dataset(
    dataset_name: str = "open-r1/OpenR1-Math-220k",
    split: str = "train",
    subset_size: int = 2000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Load and format a math dataset. Returns dicts with instruction/input/output."""
    return _load_formatted(dataset_name, split, subset_size, seed)


def load_math_dataset_with_validation(
    dataset_name: str = "open-r1/OpenR1-Math-220k",
    split: str = "train",
    subset_size: int = 2000,
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load a math dataset and split it into (train_examples, validation_examples)."""
    examples = _load_formatted(dataset_name, split, subset_size, seed)

    val_size = int(len(examples) * validation_split)
    if len(examples) > 1:
        val_size = max(1, val_size)
    train_size = len(examples) - val_size

    # Examples are already in shuffled order, so a simple slice is deterministic.
    return examples[:train_size], examples[train_size:]