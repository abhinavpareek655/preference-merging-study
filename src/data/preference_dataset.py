"""
Preference dataset loader for DPO training.

Dataset:
    trl-lib/ultrafeedback_binarized

Output format:
    {
        "prompt": [...],
        "chosen": [...],
        "rejected": [...]
    }

The loader converts the conversational UltraFeedback format into the
explicit prompt/chosen/rejected format expected by TRL's DPOTrainer.
"""

from typing import Any
import random

from datasets import Dataset, load_dataset


def _extract_conversation_parts(
    chosen: Any,
    rejected: Any,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """
    Convert a pair of conversational chosen/rejected responses into:

        prompt
        chosen_completion
        rejected_completion

    UltraFeedback stores the conversation in both chosen and rejected.
    The shared conversation history becomes the prompt, while the final
    assistant message becomes the completion.
    """

    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError(
            "Expected chosen/rejected to be conversational lists."
        )

    if len(chosen) == 0 or len(rejected) == 0:
        raise ValueError("chosen/rejected cannot be empty.")

    # Find the shared prefix.
    common_length = min(len(chosen), len(rejected))
    prefix_length = 0

    for i in range(common_length):
        c = chosen[i]
        r = rejected[i]

        if (
            isinstance(c, dict)
            and isinstance(r, dict)
            and c.get("role") == r.get("role")
            and c.get("content") == r.get("content")
        ):
            prefix_length += 1
        else:
            break

    # We expect the last message to be the assistant completion.
    chosen_last = chosen[-1]
    rejected_last = rejected[-1]

    if (
        not isinstance(chosen_last, dict)
        or not isinstance(rejected_last, dict)
    ):
        raise ValueError("Invalid message format.")

    chosen_role = chosen_last.get("role", "")
    rejected_role = rejected_last.get("role", "")

    if chosen_role != "assistant" or rejected_role != "assistant":
        raise ValueError(
            "Expected final chosen/rejected messages to be assistant messages."
        )

    prompt = chosen[:prefix_length]

    chosen_completion = [
        {
            "role": "assistant",
            "content": chosen_last.get("content", ""),
        }
    ]

    rejected_completion = [
        {
            "role": "assistant",
            "content": rejected_last.get("content", ""),
        }
    ]

    return prompt, chosen_completion, rejected_completion


def _convert_example(example: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one raw UltraFeedback example into TRL's explicit
    preference format.
    """

    chosen = example.get("chosen", [])
    rejected = example.get("rejected", [])

    prompt, chosen_completion, rejected_completion = (
        _extract_conversation_parts(chosen, rejected)
    )

    return {
        "prompt": prompt,
        "chosen": chosen_completion,
        "rejected": rejected_completion,
    }


def _is_valid(example: dict[str, Any]) -> bool:
    """Check that prompt and both completions are usable."""

    prompt = example.get("prompt", [])
    chosen = example.get("chosen", [])
    rejected = example.get("rejected", [])

    if not isinstance(prompt, list):
        return False

    if not isinstance(chosen, list) or not chosen:
        return False

    if not isinstance(rejected, list) or not rejected:
        return False

    chosen_text = chosen[-1].get("content", "").strip()
    rejected_text = rejected[-1].get("content", "").strip()

    return bool(chosen_text) and bool(rejected_text)


def load_preference_dataset(
    dataset_name: str = "trl-lib/ultrafeedback_binarized",
    train_split: str = "train",
    subset_size: int = 2000,
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load, shuffle, subset, convert, and split the preference dataset.

    Returns:
        train_dataset, validation_dataset
    """

    print(f"Loading preference dataset: {dataset_name}")

    dataset = load_dataset(
        dataset_name,
        split=train_split,
    )

    print(f"Original dataset size: {len(dataset)}")

    # Deterministic shuffle.
    dataset = dataset.shuffle(seed=seed)

    # Select requested amount.
    selected_size = min(subset_size, len(dataset))

    dataset = dataset.select(range(selected_size))

    print(f"Selected {selected_size} examples.")

    # Convert to explicit TRL preference format.
    converted = []

    for example in dataset:
        try:
            converted_example = _convert_example(example)

            if _is_valid(converted_example):
                converted.append(converted_example)

        except (TypeError, ValueError, KeyError):
            continue

    if not converted:
        raise RuntimeError(
            "No valid preference examples were produced."
        )

    # Shuffle once more deterministically after filtering.
    random.Random(seed).shuffle(converted)

    # Calculate validation size.
    validation_size = int(len(converted) * validation_split)

    if validation_size <= 0:
        raise RuntimeError(
            f"Validation split produced zero examples from {len(converted)} examples."
        )

    train_examples = converted[:-validation_size]
    validation_examples = converted[-validation_size:]

    train_dataset = Dataset.from_list(train_examples)
    validation_dataset = Dataset.from_list(validation_examples)

    return train_dataset, validation_dataset


def print_preference_examples(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    num_examples: int = 3,
) -> None:
    """Print a few examples for a dry run."""

    print("\n=== TRAINING EXAMPLES ===")

    for i in range(min(num_examples, len(train_dataset))):
        example = train_dataset[i]

        print(f"\nExample {i}:")
        print("Prompt:")
        print(example["prompt"])
        print("\nChosen:")
        print(example["chosen"])
        print("\nRejected:")
        print(example["rejected"])

    print("\n=== VALIDATION EXAMPLES ===")

    for i in range(min(num_examples, len(validation_dataset))):
        example = validation_dataset[i]

        print(f"\nExample {i}:")
        print("Prompt:")
        print(example["prompt"])
        print("\nChosen:")
        print(example["chosen"])
        print("\nRejected:")
        print(example["rejected"])