"""
Code dataset loader for MBPP (Mostly Basic Python Problems)
Loads and formats the dataset for SFT training.
"""

from datasets import load_dataset
from typing import Dict, List
import random


def load_code_dataset(
    dataset_name: str = "google-research-datasets/mbpp",
    split: str = "train",
    subset_size: int = 2000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Load and format the MBPP dataset.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        split: Dataset split to load (should be "train" for training)
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
        # Extract problem description and code from MBPP
        problem_description = example.get("text", "")
        code = example.get("code", "")

        if problem_description and code:
            formatted_examples.append({
                "instruction": problem_description,
                "input": "",  # No separate input for code problems
                "output": code,
            })

    return formatted_examples


def load_code_dataset_with_validation(
    dataset_name: str = "google-research-datasets/mbpp",
    split: str = "train",
    subset_size: int = 2000,
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load code dataset and split into train/validation sets.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        split: Dataset split to load (should be "train" for training)
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
        problem_description = example.get("text", "")
        code = example.get("code", "")

        if problem_description and code:
            formatted_examples.append({
                "instruction": problem_description,
                "input": "",
                "output": code,
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