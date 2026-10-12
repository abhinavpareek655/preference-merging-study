#!/usr/bin/env python3
"""
Reusable script for training SFT experts with LoRA (patched version).

Changes vs. the original:
  1. Bespoke-Stratos `system` prompt is kept and used in the chat template.
  2. Label masking: loss only on assistant tokens (prompt/system masked with -100).
  3. Examples longer than max_length are DROPPED (not truncated), so the model
     always sees the end-of-turn token and learns to stop.
  4. Epoch-based training + early stopping instead of a fixed 100 steps.
  5. Gradient checkpointing OFF by default; bf16 used when supported.
  6. Dynamic padding via a data collator (no padding to 512 in the dataset).

Usage:
    python scripts/train_sft_expert.py --config configs/expert_math.yaml [--dry-run] [--smoke-test]
"""

import argparse
import yaml
import os
import sys
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.math_dataset import load_math_dataset_with_validation
from data.code_dataset import load_code_dataset_with_validation
from data.instruction_dataset import load_instruction_dataset_with_validation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_instruction_dataset_direct(dataset_name, split, subset_size,
                                    validation_split, seed):
    """Load Bespoke-Stratos-17k, KEEPING the system prompt.

    Returns lists of dicts with keys: system, instruction, input, output.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)
    selected_size = min(subset_size, len(dataset))
    if selected_size == 0:
        return [], []
    dataset = dataset.select(range(selected_size))

    split_dataset = dataset.train_test_split(test_size=validation_split, seed=seed)

    def convert(example):
        # Bespoke-Stratos stores the system prompt in a top-level `system` column.
        system = example.get("system") or ""
        instruction, output = "", ""

        for message in example.get("messages") or []:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system" and not system:
                system = content
            elif role == "user" and not instruction:
                instruction = content
            elif role == "assistant" and not output:
                output = content

        # Fallback to the older `conversations` schema.
        if not instruction or not output:
            for message in example.get("conversations") or []:
                role = message.get("from", "")
                content = message.get("value", "")
                if role in ("user", "human") and not instruction:
                    instruction = content
                elif role in ("assistant", "gpt") and not output:
                    output = content

        return {"system": system, "instruction": instruction,
                "input": "", "output": output}

    def convert_all(ds):
        converted = (convert(ex) for ex in ds)  # convert each example once
        return [c for c in converted if c["instruction"] and c["output"]]

    return convert_all(split_dataset["train"]), convert_all(split_dataset["test"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_loader(expert_type):
    if expert_type == "math":
        return load_math_dataset_with_validation
    elif expert_type == "code":
        return load_code_dataset_with_validation
    elif expert_type == "instruction":
        return load_instruction_dataset_with_validation
    raise ValueError(f"Unknown expert type: {expert_type}")


def load_examples(expert_type, dataset_name, dataset_split, subset_size, seed):
    if expert_type == "instruction":
        return load_instruction_dataset_direct(
            dataset_name=dataset_name, split=dataset_split,
            subset_size=subset_size, validation_split=0.1, seed=seed)
    loader = get_dataset_loader(expert_type)
    return loader(dataset_name=dataset_name, split=dataset_split,
                  subset_size=subset_size, validation_split=0.1, seed=seed)


# ---------------------------------------------------------------------------
# Tokenization with label masking
# ---------------------------------------------------------------------------
def tokenize_example(example, tokenizer, max_length=1024):
    """Tokenize one example. Loss is computed ONLY on the assistant response.

    - system + user tokens  -> label = -100 (ignored by the loss)
    - assistant tokens      -> label = token id (includes <|im_end|>)
    - too long              -> returns None (dropped, never truncated)
    """
    user_content = example["instruction"]
    if example.get("input") and example["input"].strip():
        user_content += "\n\n" + example["input"]

    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages.append({"role": "user", "content": user_content})

    # Text up to and including "<|im_start|>assistant\n"
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    # Same prefix + response + "<|im_end|>\n"
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["output"]}],
        tokenize=False, add_generation_prompt=False)

    if not full_text.startswith(prompt_text):
        return None  # template does not have the expected prefix structure
    response_text = full_text[len(prompt_text):]

    # Tokenize the two parts separately so the boundary is always exact.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        return None  # DROP, do not truncate

    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def tokenize_dataset(examples, tokenizer, max_length, name="train"):
    """Convert a list of raw example dicts into a tokenized HF Dataset."""
    from datasets import Dataset

    rows = []
    for ex in examples:
        row = tokenize_example(ex, tokenizer, max_length)
        if row is not None:
            rows.append(row)

    dropped = len(examples) - len(rows)
    print(f"[{name}] kept {len(rows)}/{len(examples)} examples "
          f"({dropped} dropped for exceeding {max_length} tokens)")
    if not rows:
        raise RuntimeError(
            f"[{name}] no examples fit in max_length={max_length}. "
            f"Increase max_length or use a dataset with shorter outputs.")

    # Sanity check: show what is actually trained on for the first example.
    first = rows[0]
    trained = [t for t, l in zip(first["input_ids"], first["labels"]) if l != -100]
    print(f"[{name}] tokens total={len(first['input_ids'])}, "
          f"tokens with loss={len(trained)}")
    print(f"[{name}] loss text (first 200 chars): "
          f"{tokenizer.decode(trained)[:200]!r}")

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SFT expert with LoRA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)

    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    lora_config = config.get('lora', {})
    training_config = config.get('training', {})

    if args.smoke_test:
        print("Running in smoke-test mode")
        dataset_config['subset_size'] = 100
        training_config['max_steps'] = 20          # overrides epochs
        training_config['logging_steps'] = 2
        training_config['save_steps'] = 5
        training_config['eval_steps'] = 5

    model_name = model_config.get('name', "Qwen/Qwen2.5-0.5B-Instruct")
    dataset_name = dataset_config.get('name')
    subset_size = dataset_config.get('subset_size', 2000)
    dataset_split = dataset_config.get('split', "train")
    max_length = training_config.get('max_length', 1024)

    # Determine expert type
    config_file = os.path.basename(args.config)
    if 'math' in config_file:
        expert_type = "math"
    elif 'code' in config_file:
        expert_type = "code"
    elif 'instruction' in config_file:
        expert_type = "instruction"
    else:
        name = (dataset_name or "").lower()
        if "math" in name:
            expert_type = "math"
        elif "mbpp" in name or "code" in name:
            expert_type = "code"
        elif "instruction" in name or "bespoke" in name:
            expert_type = "instruction"
        else:
            raise ValueError("Config file name must contain math, code, or instruction.")
    print(f"Expert type: {expert_type}")

    # ------------------------------------------------------------------ dry run
    if args.dry_run:
        print(f"Loading dataset: {dataset_name}")
        train_examples, val_examples = load_examples(
            expert_type, dataset_name, dataset_split, subset_size, seed)
        print(f"{len(train_examples)} train / {len(val_examples)} val examples")
        for i in range(min(3, len(train_examples))):
            ex = train_examples[i]
            print(f"Example {i}:")
            print(f"  System: {ex.get('system', '')[:200]}")
            print(f"  Instruction: {ex['instruction'][:300]}")
            print(f"  Output: {ex['output'][:300]}\n")
        if not train_examples or not val_examples:
            print("DRY RUN FAILED: empty split.")
            raise SystemExit(1)
        print("DRY RUN PASSED.")
        return

    # ------------------------------------------------------------ heavy imports
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                              Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback)
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_grad_ckpt = bool(training_config.get("gradient_checkpointing", False))
    if use_grad_ckpt:
        model.config.use_cache = False

    lora_params = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    print("Applying LoRA...")
    model = get_peft_model(model, lora_params)
    if use_grad_ckpt:
        model.enable_input_require_grads()  # required for ckpt + frozen embeddings
    model.print_trainable_parameters()

    # ------------------------------------------------------------------ data
    print(f"Loading dataset: {dataset_name}")
    train_examples, val_examples = load_examples(
        expert_type, dataset_name, dataset_split, subset_size, seed)
    if not train_examples or not val_examples:
        raise RuntimeError(
            f"Dataset loading failed: {len(train_examples)} train, "
            f"{len(val_examples)} val.")
    print(f"Loaded {len(train_examples)} train / {len(val_examples)} val examples.")

    tokenized_train = tokenize_dataset(train_examples, tokenizer, max_length, "train")
    tokenized_val = tokenize_dataset(val_examples, tokenizer, max_length, "val")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)

    # -------------------------------------------------------------- training
    output_dir = training_config.get('output_dir', "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    per_device_batch = training_config.get('per_device_train_batch_size', 4)
    grad_accum = training_config.get('gradient_accumulation_steps', 2)
    print(f"Effective batch size: {per_device_batch} x {grad_accum} = "
          f"{per_device_batch * grad_accum}")

    cuda = torch.cuda.is_available()
    use_bf16 = cuda and torch.cuda.is_bf16_supported() and training_config.get("bf16", True)
    use_fp16 = cuda and not use_bf16 and training_config.get("fp16", True)
    print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    eval_steps = training_config.get("eval_steps", 50)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,

        # Epoch-based training. max_steps=-1 means "use num_train_epochs".
        num_train_epochs=training_config.get("num_train_epochs", 2),
        max_steps=training_config.get("max_steps", -1),

        learning_rate=float(training_config.get("learning_rate", 5.0e-5)),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.05),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=1.0,

        bf16=use_bf16,
        fp16=use_fp16,
        group_by_length=True,

        logging_steps=training_config.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=training_config.get("save_steps", eval_steps),
        save_total_limit=2,                     # keep only the 2 newest checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to=training_config.get("report_to", "none"),
        seed=seed,

        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=training_config.get("early_stopping_patience", 3))],
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapter (best checkpoint by eval_loss)...")
    trainer.save_model(output_dir)

    # ------------------------------------------------------- quick generation
    print("Testing generation...")
    model.config.use_cache = True
    model.eval()
    ex = val_examples[0]
    user_content = ex["instruction"]
    if ex.get("input") and ex["input"].strip():
        user_content += "\n\n" + ex["input"]
    messages = []
    if ex.get("system"):
        messages.append({"role": "system", "content": ex["system"]})
    messages.append({"role": "user", "content": user_content})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print(f"Prompt: {user_content[:300]}")
    print(f"Response: {response}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()