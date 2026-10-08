#!/usr/bin/env python3
"""
E1: Base vs E_math evaluation for the preference-merging study.

Evaluates the newly trained E_math LoRA adapter directly on top of
Qwen/Qwen2.5-0.5B-Instruct.

No LoRA baking is required.

Evaluation:
    Base model          -> GSM8K
    Base + E_math LoRA -> GSM8K

Both models are evaluated on the exact same deterministic GSM8K subset.
"""

from __future__ import annotations

import json
import os
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# IMPORTANT:
# This is the LoRA adapter repository, NOT the baked full model.
E_MATH_ADAPTER = "abhinav655/qwen25-math-expert"

GSM8K_DATASET = "openai/gsm8k"

SEED = 42

# Keep this at 200 to match your previous E1 evaluation.
MATH_EXAMPLES = 200

GEN_MAX_NEW_TOKENS_MATH = 256

RESULTS_PATH = Path("evaluations/e1_math_results.json")
PLOTS_DIR = Path("plots")
README_PATH = Path("experiments/E1_sanity/README.md")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Hugging Face token
# ============================================================

def get_hf_token() -> str | None:
    """
    Get HF token from Kaggle Secrets first, then environment.
    Public repositories can still be accessed without a token.
    """

    token = None

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")

    except Exception:
        token = os.getenv("HF_TOKEN")

    return token or None


# ============================================================
# Model utilities
# ============================================================

def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def cleanup_model(model: Any) -> None:
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model(
    model_name: str,
    token: str | None,
):
    """
    Load the original Qwen base model.
    """

    print(f"\n📦 Loading base model: {model_name}")

    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }

    if token:
        kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **kwargs,
    )

    model.eval()

    return model, tokenizer


def load_e_math_model(
    base_model_name: str,
    adapter_repo: str,
    token: str | None,
):
    """
    Load a FRESH base model and attach the E_math LoRA adapter.

    The LoRA adapter is NOT baked into the model.
    """

    print(f"\n📦 Loading fresh base model for E_math: {base_model_name}")

    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }

    if token:
        kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        **kwargs,
    )

    print(f"🔌 Loading E_math LoRA adapter: {adapter_repo}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_repo,
        token=token,
    )

    model.eval()

    print("✅ E_math LoRA adapter loaded successfully.")
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# Prompt / generation
# ============================================================

def make_user_prompt(
    tokenizer,
    prompt: str,
) -> Dict[str, Any]:
    """
    Build Qwen chat input using the tokenizer's actual chat template.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:

    encoded = make_user_prompt(
        tokenizer,
        prompt,
    )

    device = model_device(model)

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Remove the prompt tokens.
    prompt_len = encoded["input_ids"].shape[-1]

    generated = output_ids[0, prompt_len:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# GSM8K answer extraction
# ============================================================

def normalize_number(value: str) -> str | None:

    value = value.strip()

    # Remove commas and dollar signs.
    value = value.replace(",", "")
    value = value.replace("$", "")

    try:
        d = Decimal(value)

        return format(
            d.normalize(),
            "f",
        )

    except InvalidOperation:
        return None


def extract_numeric_answer(
    text: str,
) -> str | None:

    # Extract numeric values.
    matches = re.findall(
        r"[-+]?\d+(?:\.\d+)?",
        text.replace(",", ""),
    )

    if not matches:
        return None

    # GSM8K evaluation expects the final numeric answer.
    return normalize_number(matches[-1])


# ============================================================
# GSM8K evaluation
# ============================================================

def evaluate_gsm8k(
    model,
    tokenizer,
    examples,
) -> Dict[str, Any]:

    correct = 0

    predictions = []

    for index, ex in enumerate(
        tqdm(
            examples,
            desc="GSM8K evaluation",
        )
    ):

        prompt = (
            "Solve the following math problem. "
            "Show concise reasoning and end with "
            "the final numeric answer.\n\n"
            + ex["question"]
        )

        response = generate_text(
            model,
            tokenizer,
            prompt,
            GEN_MAX_NEW_TOKENS_MATH,
        )

        predicted = extract_numeric_answer(
            response
        )

        expected_text = (
            ex["answer"]
            .split("####")[-1]
            .strip()
        )

        expected = normalize_number(
            expected_text
        )

        is_correct = (
            predicted is not None
            and expected is not None
            and predicted == expected
        )

        if is_correct:
            correct += 1

        predictions.append(
            {
                "index": index,
                "question": ex["question"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "response": response,
            }
        )

    accuracy = (
        correct / len(examples)
        if examples
        else 0.0
    )

    return {
        "math_accuracy": float(accuracy),
        "correct": int(correct),
        "num_math_examples": int(len(examples)),
        "predictions": predictions,
    }


# ============================================================
# Plot
# ============================================================

def save_two_bar_plot(
    path: Path,
    title: str,
    ylabel: str,
    base_value: float,
    expert_value: float,
) -> None:

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    labels = [
        "Base",
        "E_math",
    ]

    values = [
        base_value,
        expert_value,
    ]

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.set_ylim(
        0,
        max(
            0.25,
            max(values) * 1.25,
        ),
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed(SEED)

    print("=" * 72)
    print("E1: BASE VS E_MATH EVALUATION")
    print("=" * 72)

    token = get_hf_token()

    if token:
        print(
            "🔐 Hugging Face token loaded from Kaggle Secret."
        )
    else:
        print(
            "ℹ️ No HF token found; proceeding with public Hugging Face repos."
        )

    # --------------------------------------------------------
    # Prepare output directories
    # --------------------------------------------------------

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    README_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load EXACT SAME GSM8K subset for both models
    # --------------------------------------------------------

    print("\n📚 Loading GSM8K test set...")

    gsm8k = load_dataset(
        GSM8K_DATASET,
        "main",
        split="test",
    )

    gsm8k = gsm8k.shuffle(
        seed=SEED
    ).select(
        range(
            min(
                MATH_EXAMPLES,
                len(gsm8k),
            )
        )
    )

    print(
        f"✅ GSM8K evaluation examples: {len(gsm8k)}"
    )

    # --------------------------------------------------------
    # Evaluate BASE
    # --------------------------------------------------------

    base_model, base_tokenizer = load_base_model(
        BASE_MODEL,
        token,
    )

    print("\n" + "=" * 72)
    print("Evaluating BASE model")
    print("=" * 72)

    base_results = evaluate_gsm8k(
        base_model,
        base_tokenizer,
        gsm8k,
    )

    print(
        f"\nBase GSM8K accuracy: "
        f"{base_results['math_accuracy']:.4f}"
    )

    cleanup_model(
        base_model
    )

    del base_tokenizer

    # --------------------------------------------------------
    # Evaluate E_math LoRA
    # --------------------------------------------------------

    math_model, math_tokenizer = load_e_math_model(
        BASE_MODEL,
        E_MATH_ADAPTER,
        token,
    )

    print("\n" + "=" * 72)
    print("Evaluating E_math LoRA")
    print("=" * 72)

    math_results = evaluate_gsm8k(
        math_model,
        math_tokenizer,
        gsm8k,
    )

    print(
        f"\nE_math GSM8K accuracy: "
        f"{math_results['math_accuracy']:.4f}"
    )

    cleanup_model(
        math_model
    )

    del math_tokenizer

    # --------------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------------

    base_accuracy = base_results[
        "math_accuracy"
    ]

    math_accuracy = math_results[
        "math_accuracy"
    ]

    absolute_change = (
        math_accuracy
        - base_accuracy
    )

    relative_change = (
        absolute_change / base_accuracy
        if base_accuracy != 0
        else None
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "base_model": BASE_MODEL,
        "e_math_adapter": E_MATH_ADAPTER,
        "seed": SEED,
        "dataset": GSM8K_DATASET,
        "num_examples": len(gsm8k),

        "base": {
            "math_accuracy": base_accuracy,
            "correct": base_results["correct"],
            "num_math_examples": base_results[
                "num_math_examples"
            ],
        },

        "E_math": {
            "math_accuracy": math_accuracy,
            "correct": math_results["correct"],
            "num_math_examples": math_results[
                "num_math_examples"
            ],
        },

        "comparison": {
            "absolute_change": absolute_change,
            "relative_change": relative_change,
        },

        # Keep detailed predictions separately.
        "base_predictions": base_results[
            "predictions"
        ],

        "E_math_predictions": math_results[
            "predictions"
        ],
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    save_two_bar_plot(
        PLOTS_DIR / "e1_base_vs_math.png",
        "E1: Base vs E_math",
        "GSM8K exact-match accuracy",
        base_accuracy,
        math_accuracy,
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    improvement_percentage_points = (
        absolute_change * 100
    )

    README_PATH.write_text(
        f"""# E1: Base vs E_math

## Purpose

Evaluate whether the E_math LoRA adapter improves
mathematical reasoning relative to the common base model.

## Model

Base:
`{BASE_MODEL}`

E_math LoRA:
`{E_MATH_ADAPTER}`

The E_math model is evaluated by loading the LoRA
adapter directly on top of a fresh copy of the base
model. The adapter is NOT baked for this evaluation.

## Evaluation Dataset

Dataset:
`{GSM8K_DATASET}`

Split:
`test`

Examples:
`{len(gsm8k)}`

Seed:
`{SEED}`

Both Base and E_math are evaluated on the exact same
deterministic subset.

## Results

| Model | GSM8K exact match |
|---|---:|
| Base | {base_accuracy:.4f} |
| E_math | {math_accuracy:.4f} |

## Change

Absolute change:

`{absolute_change:+.4f}`

Percentage-point change:

`{improvement_percentage_points:+.2f} pp`

Relative change:

{f"`{relative_change:+.2%}`" if relative_change is not None else "N/A"}

## Generated Files

- `evaluations/e1_math_results.json`
- `plots/e1_base_vs_math.png`

## Notes

- Generation is deterministic (`do_sample=False`).
- Both models use the Qwen tokenizer chat template.
- The same GSM8K examples are used for Base and E_math.
- E_math is evaluated directly as a LoRA adapter.
- No LoRA baking is performed during evaluation.
""",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("E1 E_MATH EVALUATION COMPLETE")
    print("=" * 72)

    print(
        f"Base GSM8K : {base_accuracy:.4f}"
    )

    print(
        f"E_math     : {math_accuracy:.4f}"
    )

    print(
        f"Change     : {absolute_change:+.4f} "
        f"({improvement_percentage_points:+.2f} pp)"
    )

    if math_accuracy > base_accuracy:
        print(
            "\n✅ E_math improved over the base model."
        )
    elif math_accuracy == base_accuracy:
        print(
            "\n⚠️ E_math matched the base model."
        )
    else:
        print(
            "\n❌ E_math performed worse than the base model."
        )

    print(
        f"\nResults: {RESULTS_PATH}"
    )

    print(
        f"Plot: {PLOTS_DIR / 'e1_base_vs_math.png'}"
    )

    print(
        f"README: {README_PATH}"
    )


if __name__ == "__main__":
    main()