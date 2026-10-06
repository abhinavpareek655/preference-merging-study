"""
Math dataset loader for OpenR1-Math-220k
Loads and formats the dataset for SFT training.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import random


def load_math_dataset(
    dataset_name: str = "open-r1/OpenR1-Math-220k",
    split: str = "train",
    subset_size: int = 2000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Load and format the OpenR1-Math-220k dataset.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        split: Dataset split to load (typically "train")
        subset_size: Number of examples to sample
        seed: Random seed for deterministic sampling

    Returns:
        List of formatted examples with keys: instruction, input, output
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Deterministically sample subset
    if subset_size < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        selected_indices = indices[:subset_size]
        dataset = dataset.select(selected_indices)

    # Format examples
    formatted_examples = []
    for example in dataset:
        # Extract question and answer from the dataset
        # Based on OpenR1-Math-220k structure
        question = example.get("problem", "")
        answer = example.get("answer", "")

        if question and answer:
            formatted_examples.append({
                "instruction": question,
                "input": "",  # No separate input for math problems
                "output": answer,
            })

    return formatted_examples


def load_math_dataset_with_validation(
    dataset_name: str = "open-r1/OpenR1-Math-220k",
    split: str = "train",
    subset_size: int = 2000,
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load math dataset and split into train/validation sets.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        split: Dataset split to load (typically "train")
        subset_size: Total number of examples to sample
        validation_split: Fraction of data to use for validation
        seed: Random seed for deterministic sampling

    Returns:
        Tuple of (train_examples, validation_examples)
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Deterministically sample subset
    if subset_size < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        selected_indices = indices[:subset_size]
        dataset = dataset.select(selected_indices)

    # Format examples
    formatted_examples = []
    for example in dataset:
        question = example.get("problem", "")
        answer = example.get("answer", "")

        if question and answer:
            formatted_examples.append({
                "instruction": question,
                "input": "",
                "output": answer,
            })

    # Split into train/validation
    val_size = int(len(formatted_examples) * validation_split)
    train_size = len(formatted_examples) - val_size

    # Deterministic shuffle for train/val split
    indices = list(range(len(formatted_examples)))
    random.Random(seed).shuffle(indices)

    train_examples = [formatted_examples[i] for i in indices[:train_size]]
    val_examples = [formatted_examples[i] for i in indices[train_size:]]

    return train_examples, val_examples