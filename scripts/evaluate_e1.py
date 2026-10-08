#!/usr/bin/env python3
"""
E1 sanity evaluation for the preference-merging study.

Kaggle-only execution. All model weights are loaded from Hugging Face.

Base:
    Qwen/Qwen2.5-0.5B-Instruct

Experts:
    abhinav655/qwen25-math-expert-full
    abhinav655/qwen25-code-expert-full
    abhinav655/qwen25-instruction-expert-full
    abhinav655/qwen25-preference-expert-full

E1 comparisons:
    Base vs E_math  -> GSM8K exact-match accuracy
    Base vs E_code  -> MBPP pass@1
    Base vs E_instr -> deterministic IFEval-supported subset score
    Base vs E_pref  -> held-out UltraFeedback pairwise accuracy + mean
                       pairwise log-probability margin

Outputs:
    evaluations/e1_results.json
    plots/e1_base_vs_math.png
    plots/e1_base_vs_code.png
    plots/e1_base_vs_instruction.png
    plots/e1_base_vs_preference_accuracy.png
    plots/e1_base_vs_preference_margin.png
    experiments/E1_sanity/README.md
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# -----------------------------
# Configuration
# -----------------------------

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EXPERT_REPOS = {
    "E_math": "abhinav655/qwen25-math-expert-full",
    # "E_code": "abhinav655/qwen25-code-expert-full",
    # "E_instr": "abhinav655/qwen25-instruction-expert-full",
    # "E_pref": "abhinav655/qwen25-preference-expert-full",
}

SEED = 42
PREF_DATASET = "trl-lib/ultrafeedback_binarized"
GSM8K_DATASET = "openai/gsm8k"
MBPP_DATASET = "google-research-datasets/mbpp"
IFEVAL_DATASET = "google/IFEval"

PREFERENCE_PAIRS = 1000  # UltraFeedback test split contains 1000 examples.
MATH_EXAMPLES = 200
MBPP_EXAMPLES = 100
IFEVAL_EXAMPLES = 100

MAX_SEQ_LENGTH = 512
GEN_MAX_NEW_TOKENS_MATH = 256
GEN_MAX_NEW_TOKENS_CODE = 256
GEN_MAX_NEW_TOKENS_INSTR = 256

PREFERENCE_BATCH_SIZE = 4
CODE_TIMEOUT_SECONDS = 5

RESULTS_PATH = Path("evaluations/e1_results.json")
PLOTS_DIR = Path("plots")
README_PATH = Path("experiments/E1_sanity/README.md")


# -----------------------------
# Reproducibility / utilities
# -----------------------------


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hf_token() -> str | None:
    """Get HF token from Kaggle Secrets first, then environment."""
    token = None
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = os.getenv("HF_TOKEN")
    return token or None


def cleanup_model(model: Any) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_model(repo_id: str, token: str | None):
    print(f"\n📦 Loading model: {repo_id}")
    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }
    if token:
        kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()
    return model, tokenizer


def make_user_prompt(tokenizer, prompt: str) -> Dict[str, Any]:
    """Build Qwen chat inputs using the tokenizer's actual chat template."""
    messages = [{"role": "user", "content": prompt}]
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
    encoded = make_user_prompt(tokenizer, prompt)
    device = model_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[-1]
    generated = output_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# -----------------------------
# Preference evaluation
# -----------------------------


def load_preference_test_set() -> List[Dict[str, Any]]:
    """Use the dataset's official test split, which is outside E_pref training."""
    ds = load_dataset(PREF_DATASET, split="test")
    if len(ds) < PREFERENCE_PAIRS:
        raise RuntimeError(
            f"Expected at least {PREFERENCE_PAIRS} preference test pairs, got {len(ds)}."
        )

    ds = ds.shuffle(seed=SEED).select(range(PREFERENCE_PAIRS))

    examples: List[Dict[str, Any]] = []
    for raw in ds:
        chosen = raw.get("chosen") or []
        rejected = raw.get("rejected") or []
        if not chosen or not rejected:
            continue
        if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
            continue

        # UltraFeedback has the same conversation prefix for chosen/rejected.
        common_len = min(len(chosen), len(rejected))
        prefix_len = 0
        for i in range(common_len - 1):
            if chosen[i] == rejected[i]:
                prefix_len += 1
            else:
                break

        prompt_messages = chosen[:prefix_len]
        chosen_response = chosen[-1].get("content", "")
        rejected_response = rejected[-1].get("content", "")

        if not prompt_messages or not chosen_response or not rejected_response:
            continue

        examples.append(
            {
                "prompt": prompt_messages,
                "chosen": [{"role": "assistant", "content": chosen_response}],
                "rejected": [{"role": "assistant", "content": rejected_response}],
            }
        )

    if len(examples) < PREFERENCE_PAIRS:
        raise RuntimeError(
            f"Only {len(examples)} valid held-out preference pairs remained; expected {PREFERENCE_PAIRS}."
        )

    print(f"✅ Held-out preference pairs: {len(examples)}")
    return examples


def _chat_text(tokenizer, messages: Sequence[Dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _sequence_logprobs_for_batch(
    model,
    tokenizer,
    prompt_messages_batch: Sequence[Sequence[Dict[str, str]]],
    response_batch: Sequence[Sequence[Dict[str, str]]],
) -> List[float]:
    """Score only completion tokens, not the prompt tokens."""
    prompt_texts = [
        _chat_text(tokenizer, p, add_generation_prompt=True)
        for p in prompt_messages_batch
    ]
    full_messages = [
        list(p) + list(r)
        for p, r in zip(prompt_messages_batch, response_batch)
    ]
    full_texts = [
        _chat_text(tokenizer, m, add_generation_prompt=False)
        for m in full_messages
    ]

    prompt_ids = [
        tokenizer(t, add_special_tokens=False)["input_ids"]
        for t in prompt_texts
    ]
    full_ids = [
        tokenizer(t, add_special_tokens=False)["input_ids"]
        for t in full_texts
    ]

    valid = [len(ids) <= MAX_SEQ_LENGTH for ids in full_ids]
    if not all(valid):
        # This function is used only on prefiltered examples; retain alignment.
        raise RuntimeError("A preference sequence exceeded MAX_SEQ_LENGTH.")

    batch = tokenizer(
        full_texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    device = model_device(model)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        outputs = model(**batch)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits[..., :-1, :], dim=-1)
    labels = batch["input_ids"][..., 1:]
    attention = batch["attention_mask"][..., 1:]
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    completion_mask = attention.clone()
    for row, prompt_len in enumerate([len(x) for x in prompt_ids]):
        keep_from = max(prompt_len - 1, 0)
        completion_mask[row, :keep_from] = 0

    seq_log_probs = (token_log_probs * completion_mask).sum(dim=1)
    return seq_log_probs.detach().float().cpu().tolist()


def evaluate_preference(model, tokenizer, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    model.eval()
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []

    filtered = []
    for ex in examples:
        chosen_full = list(ex["prompt"]) + list(ex["chosen"])
        rejected_full = list(ex["prompt"]) + list(ex["rejected"])
        chosen_text = _chat_text(tokenizer, chosen_full, add_generation_prompt=False)
        rejected_text = _chat_text(tokenizer, rejected_full, add_generation_prompt=False)
        if len(tokenizer(chosen_text, add_special_tokens=False)["input_ids"]) <= MAX_SEQ_LENGTH and len(
            tokenizer(rejected_text, add_special_tokens=False)["input_ids"]
        ) <= MAX_SEQ_LENGTH:
            filtered.append(ex)

    if not filtered:
        raise RuntimeError("No preference examples fit within MAX_SEQ_LENGTH.")

    if len(filtered) != len(examples):
        print(f"⚠️ Preference examples truncated out by max length: {len(examples) - len(filtered)}")

    for start in tqdm(range(0, len(filtered), PREFERENCE_BATCH_SIZE), desc="Preference evaluation"):
        batch = filtered[start : start + PREFERENCE_BATCH_SIZE]
        prompts = [x["prompt"] for x in batch]
        chosen = [x["chosen"] for x in batch]
        rejected = [x["rejected"] for x in batch]

        chosen_scores.extend(_sequence_logprobs_for_batch(model, tokenizer, prompts, chosen))
        rejected_scores.extend(_sequence_logprobs_for_batch(model, tokenizer, prompts, rejected))

    margins = np.asarray(chosen_scores) - np.asarray(rejected_scores)
    accuracy = float(np.mean(margins > 0))

    return {
        "preference_accuracy": accuracy,
        "mean_pairwise_logprob_margin": float(np.mean(margins)),
        "num_pairs_evaluated": int(len(margins)),
    }


# -----------------------------
# GSM8K evaluation
# -----------------------------


def normalize_number(value: str) -> str | None:
    value = value.strip().replace(",", "")
    value = value.replace("$", "")
    try:
        d = Decimal(value)
        return format(d.normalize(), "f")
    except InvalidOperation:
        return None


def extract_numeric_answer(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return normalize_number(matches[-1])


def evaluate_gsm8k(model, tokenizer) -> Dict[str, Any]:
    ds = load_dataset(GSM8K_DATASET, "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(MATH_EXAMPLES, len(ds))))

    correct = 0
    for ex in tqdm(ds, desc="GSM8K evaluation"):
        prompt = (
            "Solve the following math problem. Show concise reasoning and end with the final numeric answer.\n\n"
            + ex["question"]
        )
        response = generate_text(model, tokenizer, prompt, GEN_MAX_NEW_TOKENS_MATH)
        predicted = extract_numeric_answer(response)
        expected_text = ex["answer"].split("####")[-1].strip()
        expected = normalize_number(expected_text)
        if predicted is not None and expected is not None and predicted == expected:
            correct += 1

    accuracy = correct / len(ds) if ds else 0.0
    return {
        "math_accuracy": float(accuracy),
        "num_math_examples": int(len(ds)),
    }


# -----------------------------
# MBPP pass@1 evaluation
# -----------------------------


def extract_python_code(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def run_mbpp_tests(code: str, setup: str, tests: Sequence[str], timeout: int = CODE_TIMEOUT_SECONDS) -> bool:
    """Run generated Python code against the provided MBPP tests in a child process."""
    payload = "\n".join(
        part for part in [setup.strip(), code.strip(), "\n".join(tests)] if part
    ) + "\n"

    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as td:
        script = Path(td) / "solution.py"
        script.write_text(payload, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


def evaluate_mbpp(model, tokenizer) -> Dict[str, Any]:
    ds = load_dataset(MBPP_DATASET, split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(MBPP_EXAMPLES, len(ds))))

    correct = 0
    for ex in tqdm(ds, desc="MBPP pass@1 evaluation"):
        prompt = (
            "Write only the Python code that solves the following programming problem. "
            "Do not include markdown fences or explanation.\n\n"
            + ex["text"]
        )
        response = generate_text(model, tokenizer, prompt, GEN_MAX_NEW_TOKENS_CODE)
        code = extract_python_code(response)
        if run_mbpp_tests(
            code,
            ex.get("test_setup_code", ""),
            ex.get("test_list", []),
        ):
            correct += 1

    pass1 = correct / len(ds) if ds else 0.0
    return {
        "mbpp_pass_at_1": float(pass1),
        "num_code_examples": int(len(ds)),
    }


# -----------------------------
# Deterministic IFEval-supported subset
# -----------------------------

SUPPORTED_IFEVAL_IDS = {
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "punctuation:no_comma",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:json_format",
    "detectable_format:number_bullet_lists",
    "detectable_format:title",
    "detectable_format:number_highlighted_sections",
    "startend:quotation",
    "combination:repeat_prompt",
}


def sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if p.strip()]
    return len(parts)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def bullet_count(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[*•-]|\d+[.)])\s+", text))


def highlighted_section_count(text: str) -> int:
    return len(re.findall(r"(?<!\*)\*[^*\n]+\*(?!\*)", text))


def relation_ok(value: int, relation: str | None, target: int) -> bool:
    if relation == "at least":
        return value >= target
    if relation == "at most":
        return value <= target
    if relation == "less than":
        return value < target
    if relation == "greater than":
        return value > target
    if relation in ("exactly", "equal to"):
        return value == target
    return value == target


def evaluate_ifeval_example(prompt: str, instruction_ids: Sequence[str], kwargs_list: Sequence[Dict[str, Any]], response: str) -> bool:
    for instruction_id, kw in zip(instruction_ids, kwargs_list):
        kw = kw or {}
        if instruction_id == "keywords:existence":
            if not all(str(k).lower() in response.lower() for k in kw.get("keywords", [])):
                return False
        elif instruction_id == "keywords:frequency":
            keyword = str(kw.get("keyword", ""))
            required = int(kw.get("frequency", 0))
            if response.lower().count(keyword.lower()) < required:
                return False
        elif instruction_id == "keywords:forbidden_words":
            if any(str(k).lower() in response.lower() for k in kw.get("forbidden_words", [])):
                return False
        elif instruction_id == "punctuation:no_comma":
            if "," in response:
                return False
        elif instruction_id == "change_case:english_lowercase":
            if response != response.lower():
                return False
        elif instruction_id == "change_case:english_capital":
            if response != response.upper():
                return False
        elif instruction_id == "length_constraints:number_words":
            if not relation_ok(word_count(response), kw.get("relation"), int(kw.get("num_words", 0))):
                return False
        elif instruction_id == "length_constraints:number_sentences":
            if not relation_ok(sentence_count(response), kw.get("relation"), int(kw.get("num_sentences", 0))):
                return False
        elif instruction_id == "detectable_content:number_placeholders":
            count = len(re.findall(r"\[[^\[\]\n]+\]", response))
            if not relation_ok(count, "at least", int(kw.get("num_placeholders", 0))):
                return False
        elif instruction_id == "detectable_content:postscript":
            marker = str(kw.get("postscript_marker", ""))
            if marker not in response:
                return False
        elif instruction_id == "detectable_format:json_format":
            try:
                json.loads(response.strip())
            except Exception:
                return False
        elif instruction_id == "detectable_format:number_bullet_lists":
            if bullet_count(response) != int(kw.get("num_bullets", 0)):
                return False
        elif instruction_id == "detectable_format:title":
            if not re.search(r"<<[^>]+>>", response):
                return False
        elif instruction_id == "detectable_format:number_highlighted_sections":
            if highlighted_section_count(response) != int(kw.get("num_highlights", 0)):
                return False
        elif instruction_id == "startend:quotation":
            stripped = response.strip()
            if not (stripped.startswith('"') and stripped.endswith('"')):
                return False
        elif instruction_id == "combination:repeat_prompt":
            repeated = str(kw.get("prompt_to_repeat", prompt))
            if not response.startswith(repeated):
                return False
        else:
            return False
    return True


def load_ifeval_subset() -> List[Dict[str, Any]]:
    ds = load_dataset(IFEVAL_DATASET, split="train")
    supported = []
    for ex in ds:
        ids = ex["instruction_id_list"]
        if all(i in SUPPORTED_IFEVAL_IDS for i in ids):
            supported.append(ex)
    rng = random.Random(SEED)
    rng.shuffle(supported)
    selected = supported[: min(IFEVAL_EXAMPLES, len(supported))]
    if not selected:
        raise RuntimeError("No supported IFEval examples found.")
    print(f"✅ IFEval-supported subset: {len(selected)} examples")
    return selected


def evaluate_ifeval(model, tokenizer, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    correct = 0
    for ex in tqdm(examples, desc="Instruction-following evaluation"):
        response = generate_text(model, tokenizer, ex["prompt"], GEN_MAX_NEW_TOKENS_INSTR)
        if evaluate_ifeval_example(
            ex["prompt"],
            ex["instruction_id_list"],
            ex["kwargs"],
            response,
        ):
            correct += 1
    accuracy = correct / len(examples) if examples else 0.0
    return {
        "instruction_following_accuracy": float(accuracy),
        "num_instruction_examples": int(len(examples)),
    }


# -----------------------------
# Plotting
# -----------------------------


def save_two_bar_plot(path: Path, title: str, ylabel: str, base_value: float, expert_value: float, expert_name: str, percent: bool = True) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Base", expert_name]
    values = [base_value, expert_value]
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.0f}%")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main evaluation
# -----------------------------


def main() -> None:
    set_seed(SEED)

    token = get_hf_token()
    if token:
        print("🔐 Hugging Face token loaded from Kaggle Secret.")
    else:
        print("ℹ️ No HF token found; proceeding with public Hugging Face repos.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load evaluation sets once.
    preference_examples = load_preference_test_set()
    ifeval_examples = load_ifeval_subset()

    results: Dict[str, Any] = {}

    # Evaluate base ONCE on all E1 metrics.
    base_model, base_tokenizer = load_model(BASE_MODEL, token)
    results["base"] = {}
    results["base"].update(evaluate_preference(base_model, base_tokenizer, preference_examples))
    results["base"].update(evaluate_gsm8k(base_model, base_tokenizer))
    results["base"].update(evaluate_mbpp(base_model, base_tokenizer))
    results["base"].update(evaluate_ifeval(base_model, base_tokenizer, ifeval_examples))
    cleanup_model(base_model)

    # Evaluate each expert only on the capability needed for E1.
    for expert_name, repo in EXPERT_REPOS.items():
        model, tokenizer = load_model(repo, token)
        results[expert_name] = {}

        if expert_name == "E_math":
            results[expert_name].update(evaluate_gsm8k(model, tokenizer))
#         elif expert_name == "E_code":
#             results[expert_name].update(evaluate_mbpp(model, tokenizer))
#         elif expert_name == "E_instr":
#             results[expert_name].update(evaluate_ifeval(model, tokenizer, ifeval_examples))
#         elif expert_name == "E_pref":
#             results[expert_name].update(evaluate_preference(model, tokenizer, preference_examples))

        cleanup_model(model)

    # Save raw results.
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Separate Base-vs-Expert plots.
    save_two_bar_plot(
        PLOTS_DIR / "e1_base_vs_math.png",
        "E1: Base vs E_math",
        "GSM8K exact-match accuracy",
        results["base"]["math_accuracy"],
        results["E_math"]["math_accuracy"],
        "E_math",
    )

    # save_two_bar_plot(
    #     PLOTS_DIR / "e1_base_vs_code.png",
    #     "E1: Base vs E_code",
    #     "MBPP pass@1",
    #     results["base"]["mbpp_pass_at_1"],
    #     results["E_code"]["mbpp_pass_at_1"],
    #     "E_code",
    # )

    # save_two_bar_plot(
    #     PLOTS_DIR / "e1_base_vs_instruction.png",
    #     "E1: Base vs E_instr",
    #     "Instruction-following accuracy",
    #     results["base"]["instruction_following_accuracy"],
    #     results["E_instr"]["instruction_following_accuracy"],
    #     "E_instr",
    # )

    # save_two_bar_plot(
    #     PLOTS_DIR / "e1_base_vs_preference_accuracy.png",
    #     "E1: Base vs E_pref",
    #     "Held-out preference accuracy",
    #     results["base"]["preference_accuracy"],
    #     results["E_pref"]["preference_accuracy"],
    #     "E_pref",
    # )

    # Margin is not a bounded percentage, so use a separate plot with the true scale.
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(7, 5))
    # labels = ["Base", "E_pref"]
    # values = [
    #     results["base"]["mean_pairwise_logprob_margin"],
    #     results["E_pref"]["mean_pairwise_logprob_margin"],
    # ]
    # bars = ax.bar(labels, values)
    # ax.axhline(0.0, linewidth=1)
    # ax.set_title("E1: Base vs E_pref — pairwise log-probability margin")
    # ax.set_ylabel("Mean chosen − rejected log probability")
    # for bar, value in zip(bars, values):
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         value,
    #         f"{value:.3f}",
    #         ha="center",
    #         va="bottom" if value >= 0 else "top",
    #     )
    # fig.tight_layout()
    # fig.savefig(PLOTS_DIR / "e1_base_vs_preference_margin.png", dpi=180, bbox_inches="tight")
    # plt.close(fig)

    # README.
    README_PATH.write_text(
        f"""# E1 Sanity Evaluation\n\n"
        f"## Purpose\n"
        f"Verify that each specialist improves its intended capability and that E_pref improves held-out preference metrics versus the common base.\n\n"
        f"## Model\n"
        f"Qwen/Qwen2.5-0.5B-Instruct\n\n"
        f"## Held-out evaluation sets\n"
        f"- UltraFeedback test split: {results['base']['num_pairs_evaluated']} pairs\n"
        f"- GSM8K deterministic subset: {results['base']['num_math_examples']} examples\n"
        f"- MBPP deterministic subset: {results['base']['num_code_examples']} examples\n"
        f"- IFEval-supported deterministic subset: {results['base']['num_instruction_examples']} examples\n\n"
        f"## Results\n\n"
        f"| Comparison | Base | Expert | Metric |\n"
        f"|---|---:|---:|---|\n"
        f"| Base vs E_math | {results['base']['math_accuracy']:.4f} | {results['E_math']['math_accuracy']:.4f} | GSM8K exact match |\n"
        # f"| Base vs E_code | {results['base']['mbpp_pass_at_1']:.4f} | {results['E_code']['mbpp_pass_at_1']:.4f} | MBPP pass@1 |\n"
        # f"| Base vs E_instr | {results['base']['instruction_following_accuracy']:.4f} | {results['E_instr']['instruction_following_accuracy']:.4f} | IFEval-supported subset accuracy |\n"
        # f"| Base vs E_pref | {results['base']['preference_accuracy']:.4f} | {results['E_pref']['preference_accuracy']:.4f} | Held-out preference accuracy |\n"
        # f"| Base vs E_pref | {results['base']['mean_pairwise_logprob_margin']:.4f} | {results['E_pref']['mean_pairwise_logprob_margin']:.4f} | Mean pairwise log-probability margin |\n\n"
        f"Seed: {SEED}\n\n"
        f"## Generated files\n"
        f"- evaluations/e1_results.json\n"
        f"- plots/e1_base_vs_math.png\n"
        # f"- plots/e1_base_vs_code.png\n"
        # f"- plots/e1_base_vs_instruction.png\n"
        # f"- plots/e1_base_vs_preference_accuracy.png\n"
        # f"- plots/e1_base_vs_preference_margin.png\n\n"
        f"## Notes\n"
        f"- All model weights are loaded from Hugging Face.\n"
        f"- All evaluation is intended to run on Kaggle only.\n"
        f"- MBPP is evaluated as true pass@1 by executing one generated solution against the provided test cases.\n"
        f"- The instruction metric is a deterministic supported subset of the official Google IFEval dataset, not the full IFEval evaluator.\n"
        f""",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("E1 SANITY EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Base vs E_math       : {results['base']['math_accuracy']:.4f} -> {results['E_math']['math_accuracy']:.4f}")
    # print(f"Base vs E_code       : {results['base']['mbpp_pass_at_1']:.4f} -> {results['E_code']['mbpp_pass_at_1']:.4f}")
    # print(f"Base vs E_instr      : {results['base']['instruction_following_accuracy']:.4f} -> {results['E_instr']['instruction_following_accuracy']:.4f}")
    # print(f"Base vs E_pref       : {results['base']['preference_accuracy']:.4f} -> {results['E_pref']['preference_accuracy']:.4f}")
    # print(f"Preference margin    : {results['base']['mean_pairwise_logprob_margin']:.4f} -> {results['E_pref']['mean_pairwise_logprob_margin']:.4f}")
    print(f"\nResults: {RESULTS_PATH}")
    print(f"Plots  : {PLOTS_DIR}/")
    print(f"README : {README_PATH}")


if __name__ == "__main__":
    main()