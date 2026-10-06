#!/usr/bin/env python3

"""
Train the preference expert (E_pref) using DPO + LoRA.

Usage:

    python scripts/train_dpo_expert.py \
        --config configs/expert_preference.yaml \
        --dry-run

    python scripts/train_dpo_expert.py \
        --config configs/expert_preference.yaml

    python scripts/train_dpo_expert.py \
        --config configs/expert_preference.yaml \
        --smoke-test
"""

import argparse
import os
import random
import sys

import numpy as np
import yaml


# Allow importing from src/
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from data.preference_dataset import (
    load_preference_dataset,
    print_preference_examples,
)


def set_seed(seed: int) -> None:
    """Set Python and NumPy seeds."""

    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train E_pref using DPO + LoRA."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to DPO configuration YAML file.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset and print examples without training.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small 100-example / 20-step training test.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    config = load_config(args.config)

    print(f"Loaded config from: {args.config}")

    # ---------------------------------------------------------
    # Configuration sections
    # ---------------------------------------------------------

    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    lora_config = config.get("lora", {})
    dpo_config = config.get("dpo", {})
    training_config = config.get("training", {})

    seed = training_config.get("seed", 42)

    set_seed(seed)

    print(f"Set seed to: {seed}")
    print("Expert type: preference")

    # ---------------------------------------------------------
    # Smoke-test overrides
    # ---------------------------------------------------------

    if args.smoke_test:

        print(
            "Running in smoke-test mode: "
            "100 examples and 20 training steps."
        )

        dataset_config["subset_size"] = 100

        training_config["max_steps"] = 20
        training_config["logging_steps"] = 2
        training_config["eval_steps"] = 5
        training_config["save_steps"] = 5

    # ---------------------------------------------------------
    # Extract configuration
    # ---------------------------------------------------------

    model_name = model_config.get(
        "name",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )

    dataset_name = dataset_config.get(
        "name",
        "trl-lib/ultrafeedback_binarized",
    )

    train_split = dataset_config.get(
        "train_split",
        "train",
    )

    subset_size = dataset_config.get(
        "subset_size",
        2000,
    )

    validation_split = dataset_config.get(
        "validation_split",
        0.1,
    )

    print(f"Model: {model_name}")
    print(f"Preference dataset: {dataset_name}")

    # ---------------------------------------------------------
    # Load preference data
    # ---------------------------------------------------------

    train_dataset, eval_dataset = load_preference_dataset(
        dataset_name=dataset_name,
        train_split=train_split,
        subset_size=subset_size,
        validation_split=validation_split,
        seed=seed,
    )

    print(
        f"Loaded {len(train_dataset)} training examples "
        f"and {len(eval_dataset)} validation examples."
    )

    # ---------------------------------------------------------
    # Dry run
    # ---------------------------------------------------------

    if args.dry_run:

        print_preference_examples(
            train_dataset,
            eval_dataset,
            num_examples=3,
        )

        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            raise RuntimeError(
                "DRY RUN FAILED: empty train or validation dataset."
            )

        print("\nDRY RUN PASSED.")
        return

    # ---------------------------------------------------------
    # Heavy ML imports
    # ---------------------------------------------------------

    import torch
    from transformers import AutoTokenizer

    from peft import LoraConfig

    from trl import DPOConfig, DPOTrainer

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_config.get("revision", "main"),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # LoRA configuration
    # ---------------------------------------------------------

    lora_params = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "v_proj"],
        ),
        lora_dropout=lora_config.get(
            "lora_dropout",
            0.05,
        ),
        bias=lora_config.get(
            "bias",
            "none",
        ),
        task_type=lora_config.get(
            "task_type",
            "CAUSAL_LM",
        ),
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    output_dir = training_config.get(
        "output_dir",
        "./outputs/E_pref",
    )

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # DPO training configuration
    # ---------------------------------------------------------

    training_args = DPOConfig(
        output_dir=output_dir,

        # DPO hyperparameters
        beta=float(
            dpo_config.get("beta", 0.1)
        ),

        loss_type=dpo_config.get(
            "loss_type",
            "sigmoid",
        ),

        max_length=int(
            dpo_config.get("max_length", 512)
        ),

        max_length=int(
            dpo_config.get("max_length", 256)
        ),

        # Training batch
        per_device_train_batch_size=int(
            training_config.get(
                "per_device_train_batch_size",
                2,
            )
        ),

        per_device_eval_batch_size=int(
            training_config.get(
                "per_device_eval_batch_size",
                2,
            )
        ),

        gradient_accumulation_steps=int(
            training_config.get(
                "gradient_accumulation_steps",
                4,
            )
        ),

        # Optimization
        warmup_steps=int(
            training_config.get(
                "warmup_steps",
                10,
            )
        ),

        max_steps=int(
            training_config.get(
                "max_steps",
                100,
            )
        ),

        learning_rate=float(
            training_config.get(
                "learning_rate",
                5e-5,
            )
        ),

        # Mixed precision
        fp16=bool(
            training_config.get(
                "fp16",
                True,
            )
        ),

        # Logging / evaluation
        logging_steps=int(
            training_config.get(
                "logging_steps",
                5,
            )
        ),

        eval_strategy=training_config.get(
            "eval_strategy",
            "steps",
        ),

        eval_steps=int(
            training_config.get(
                "eval_steps",
                20,
            )
        ),

        # Saving
        save_strategy=training_config.get(
            "save_strategy",
            "steps",
        ),

        save_steps=int(
            training_config.get(
                "save_steps",
                20,
            )
        ),

        # Reproducibility
        seed=seed,

        # Disable external logging
        report_to=training_config.get(
            "report_to",
            "none",
        ),

        # Keep evaluation columns
        remove_unused_columns=False,
    )

    # ---------------------------------------------------------
    # Create DPO trainer
    # ---------------------------------------------------------

    print("Initializing DPOTrainer...")

    trainer = DPOTrainer(
        model=model_name,

        # DPOTrainer automatically uses the initial model
        # as the reference policy when ref_model is omitted.
        ref_model=None,

        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        processing_class=tokenizer,

        peft_config=lora_params,
    )

    # ---------------------------------------------------------
    # Print LoRA parameters
    # ---------------------------------------------------------

    if hasattr(
        trainer.model,
        "print_trainable_parameters",
    ):
        trainer.model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nStarting DPO training...")

    trainer.train()

    # ---------------------------------------------------------
    # Save adapter
    # ---------------------------------------------------------

    print("\nSaving DPO LoRA adapter...")

    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    print("\nTraining completed successfully.")

    print(f"Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()