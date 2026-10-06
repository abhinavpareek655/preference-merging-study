"""Dataset loader for HuggingFaceH4/Bespoke-Stratos-17k."""

from datasets import load_dataset


def load_instruction_dataset_with_validation(
    dataset_name: str,
    split: str = "train",
    subset_size: int = 2000,
    validation_split: float = 0.1,
    seed: int = 42,
):
    """Load Bespoke-Stratos-17k and normalize it to instruction/input/output.

    The HuggingFaceH4 version exposes `messages` and `conversations`.
    We use the first user message as the instruction and the first assistant
    message as the target output.
    """
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)

    selected_size = min(subset_size, len(dataset))
    dataset = dataset.select(range(selected_size))

    if selected_size == 0:
        return [], []

    split_dataset = dataset.train_test_split(
        test_size=validation_split,
        seed=seed,
    )

    def convert(example):
        instruction = ""
        output = ""

        for message in example.get("messages") or []:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user" and not instruction:
                instruction = content
            elif role == "assistant" and not output:
                output = content

        if not instruction or not output:
            for message in example.get("conversations") or []:
                role = message.get("from", "")
                content = message.get("value", "")
                if role == "user" and not instruction:
                    instruction = content
                elif role in ("assistant", "gpt") and not output:
                    output = content

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
        }

    def valid(example):
        return bool(example["instruction"].strip()) and bool(example["output"].strip())

    train_examples = [x for x in map(convert, split_dataset["train"]) if valid(x)]
    val_examples = [x for x in map(convert, split_dataset["test"]) if valid(x)]

    return train_examples, val_examples
