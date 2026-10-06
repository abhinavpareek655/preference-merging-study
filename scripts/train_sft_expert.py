#!/usr/bin/env python3
"""
Reusable script for training SFT experts with LoRA.

Usage:
    python scripts/train_sft_expert.py --config <config_file> [--dry-run]

Example:
    python scripts/train_sft_expert.py --config configs/expert_math.yaml --dry-run
    python scripts/train_sft_expert.py --config configs/expert_math.yaml
"""

import argparse
import yaml
import os
import sys
import random
import numpy as np

# Import dataset loaders
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.math_dataset import load_math_dataset_with_validation
from data.code_dataset import load_code_dataset_with_validation
from data.instruction_dataset import load_instruction_dataset_with_validation


def load_instruction_dataset_direct(dataset_name: str, split: str, subset_size: int,
                                    validation_split: float, seed: int):
    """Load Bespoke-Stratos-17k directly from its TRL-compatible schema.

    The dataset contains `system`, `conversations`, and `messages` fields.
    We use the user message as the instruction and the assistant message as
    the target output.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    # Shuffle deterministically, then cap the requested number of examples.
    dataset = dataset.shuffle(seed=seed)
    selected_size = min(subset_size, len(dataset))
    dataset = dataset.select(range(selected_size))

    if selected_size == 0:
        return [], []

    # Create a deterministic 90/10 train/validation split.
    split_dataset = dataset.train_test_split(
        test_size=validation_split,
        seed=seed,
    )

    def convert(example):
        messages = example.get("messages") or []

        instruction = ""
        output = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user" and not instruction:
                instruction = content
            elif role == "assistant" and not output:
                output = content

        # Fallback to the older `conversations` representation if needed.
        if not instruction or not output:
            conversations = example.get("conversations") or []
            for message in conversations:
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

    train_examples = [
        convert(example)
        for example in split_dataset["train"]
        if convert(example)["instruction"] and convert(example)["output"]
    ]
    val_examples = [
        convert(example)
        for example in split_dataset["test"]
        if convert(example)["instruction"] and convert(example)["output"]
    ]

    return train_examples, val_examples


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset_loader(expert_type: str):
    """Return the appropriate dataset loader function."""
    if expert_type == "math":
        return load_math_dataset_with_validation
    elif expert_type == "code":
        return load_code_dataset_with_validation
    elif expert_type == "instruction":
        return load_instruction_dataset_with_validation
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


def main():
    parser = argparse.ArgumentParser(description="Train SFT expert with LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and show examples, do not train")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with reduced dataset size and steps")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Set seed
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    print(f"Set seed to: {seed}")

    # Extract configuration
    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    lora_config = config.get('lora', {})
    training_config = config.get('training', {})

    # Apply smoke-test settings if requested
    if args.smoke_test:
        print("Running in smoke-test mode: reducing dataset size and training steps")
        dataset_config['subset_size'] = 100
        training_config['max_steps'] = 20
        # Also reduce logging and save frequency for smoke test
        training_config['logging_steps'] = 2
        training_config['save_steps'] = 5
        training_config['eval_steps'] = 5

    model_name = model_config.get('name', "Qwen/Qwen2.5-0.5B-Instruct")
    dataset_name = dataset_config.get('name')
    subset_size = dataset_config.get('subset_size', 2000)
    dataset_split = dataset_config.get('split', "train")

    # Determine expert type from config path or config content
    config_file = os.path.basename(args.config)
    if 'math' in config_file:
        expert_type = "math"
    elif 'code' in config_file:
        expert_type = "code"
    elif 'instruction' in config_file:
        expert_type = "instruction"
    else:
        # Try to infer from dataset name
        if dataset_name and "math" in dataset_name.lower():
            expert_type = "math"
        elif dataset_name and ("mbpp" in dataset_name.lower() or "code" in dataset_name.lower()):
            expert_type = "code"
        elif dataset_name and ("instruction" in dataset_name.lower() or "bespoke" in dataset_name.lower()):
            expert_type = "instruction"
        else:
            raise ValueError("Could not determine expert type from config. Please ensure config file name contains math, code, or instruction.")

    print(f"Expert type: {expert_type}")

    if args.dry_run:
        # For dry-run, we only need to load the dataset and show examples
        print(f"Loading dataset: {dataset_name}")
        if expert_type == "instruction":
            # Bespoke-Stratos-17k has a conversations/messages schema, so
            # load it directly instead of relying on the older generic loader.
            train_examples, val_examples = load_instruction_dataset_direct(
                dataset_name=dataset_name,
                split=dataset_split,
                subset_size=subset_size,
                validation_split=0.1,
                seed=seed,
            )
        else:
            loader = get_dataset_loader(expert_type)
            train_examples, val_examples = loader(
                dataset_name=dataset_name,
                split=dataset_split,
                subset_size=subset_size,
                validation_split=0.1,
                seed=seed,
            )

        actual_total = len(train_examples) + len(val_examples)
        print(
            f"Requested up to {subset_size} examples; "
            f"selected {actual_total} usable examples."
        )
        print(
            f"Loaded {len(train_examples)} training examples and "
            f"{len(val_examples)} validation examples."
        )

        print("\n=== DRY RUN ===")
        print("Showing first 3 training examples:")
        for i in range(min(3, len(train_examples))):
            print(f"Example {i}:")
            print(f"  Instruction: {train_examples[i]['instruction']}")
            print(f"  Input: {train_examples[i]['input']}")
            print(f"  Output: {train_examples[i]['output']}")
            print()

        print("Showing first 3 validation examples:")
        for i in range(min(3, len(val_examples))):
            print(f"Example {i}:")
            print(f"  Instruction: {val_examples[i]['instruction']}")
            print(f"  Input: {val_examples[i]['input']}")
            print(f"  Output: {val_examples[i]['output']}")
            print()

        if len(train_examples) == 0 or len(val_examples) == 0:
            print("\nDRY RUN FAILED: one or more splits contain zero usable examples.")
            raise SystemExit(1)

        print("\nDRY RUN PASSED: dataset contains usable train and validation examples.")
        print("Dry run completed. Exiting.")
        return

    # If not dry-run, we need the heavy libraries
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    # Set seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # torch_dtype will be set by TrainingArguments if fp16 is enabled
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # LoRA configuration
    lora_params = LoraConfig(
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        bias=lora_config.get('bias', "none"),
        task_type=lora_config.get('task_type', "CAUSAL_LM"),
    )

    # Prepare model for LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_params)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    if expert_type == "instruction":
        train_examples, val_examples = load_instruction_dataset_direct(
            dataset_name=dataset_name,
            split=dataset_split,
            subset_size=subset_size,
            validation_split=0.1,
            seed=seed,
        )
    else:
        loader = get_dataset_loader(expert_type)
        train_examples, val_examples = loader(
            dataset_name=dataset_name,
            split=dataset_split,
            subset_size=subset_size,
            validation_split=0.1,
            seed=seed,
        )

    if len(train_examples) == 0 or len(val_examples) == 0:
        raise RuntimeError(
            f"Dataset loading failed: {len(train_examples)} train, "
            f"{len(val_examples)} validation examples."
        )

    actual_total = len(train_examples) + len(val_examples)
    print(
        f"Requested up to {subset_size} examples; "
        f"selected {actual_total} usable examples."
    )
    print(
        f"Loaded {len(train_examples)} training examples and "
        f"{len(val_examples)} validation examples."
    )

    # Format examples
    def format_example(example, tokenizer):
        # Format as a conversation
        messages = [
            {"role": "user", "content": example["instruction"] + (" " + example["input"] if example["input"] else "")},
            {"role": "assistant", "content": example["output"]},
        ]
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # We are training on the full sequence
        )
        return {"text": text}

    def format_examples(examples):
        formatted = [format_example(ex, tokenizer) for ex in examples]
        return Dataset.from_list(formatted)

    train_dataset = format_examples(train_examples)
    val_dataset = format_examples(val_examples)

    # Tokenize datasets
    def tokenize_function(examples, tokenizer):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    def tokenize_dataset(dataset, tokenizer):
        return dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=["text"]
        )

    tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val_dataset = tokenize_dataset(val_dataset, tokenizer)

    # Set up training arguments
    output_dir = training_config.get('output_dir', "./outputs")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
        warmup_steps=training_config.get('warmup_steps', 10),
        max_steps=training_config.get('max_steps', 100),
        learning_rate=training_config.get('learning_rate', 1e-4),
        fp16=training_config.get('fp16', True),
        logging_steps=training_config.get('logging_steps', 5),
        save_steps=training_config.get('save_steps', 20),
        eval_steps=training_config.get('eval_steps', 20),
        report_to=training_config.get('report_to', "none"),
        seed=seed,
        load_best_model_at_end=True,  # Load best model at end
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    print("Saving LoRA adapter...")
    trainer.save_model(output_dir)

    # Optional: Test generation with the fine-tuned model
    print("Testing generation...")
    model.eval()
    # Use a sample from the validation set for testing
    if len(val_examples) > 0:
        test_example = val_examples[0]
        test_instruction = test_example["instruction"]
        test_input = test_example["input"]
        full_prompt = test_instruction + (" " + test_input if test_input else "")
        messages = [
            {"role": "user", "content": full_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {full_prompt}")
        print(f"Response: {response}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
