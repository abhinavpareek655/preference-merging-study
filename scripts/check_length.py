#!/usr/bin/env python3
"""
Count how many training examples exceed a token length.

Usage (run from the same folder as train_sft_expert.py):
    python check_lengths.py --config configs/expert_math.yaml
    python check_lengths.py --config configs/expert_math.yaml --limit 1024
"""
import argparse
import os
import numpy as np
from transformers import AutoTokenizer

from train_sft_expert import load_config, load_examples


def detect_expert_type(config_path, dataset_name):
    f = os.path.basename(config_path)
    for t in ("math", "code", "instruction"):
        if t in f:
            return t
    n = (dataset_name or "").lower()
    if "math" in n:
        return "math"
    if "mbpp" in n or "code" in n:
        return "code"
    return "instruction"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=1024)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    seed = cfg.get("training", {}).get("seed", 42)
    expert_type = detect_expert_type(args.config, ds["name"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    train, val = load_examples(expert_type, ds["name"], ds.get("split", "train"),
                               ds.get("subset_size", 2000), seed)
    examples = train + val

    lengths = []
    for ex in examples:
        user = ex["instruction"]
        if ex.get("input") and ex["input"].strip():
            user += "\n\n" + ex["input"]
        msgs = []
        if ex.get("system"):
            msgs.append({"role": "system", "content": ex["system"]})
        msgs += [{"role": "user", "content": user},
                 {"role": "assistant", "content": ex["output"]}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))

    lengths = np.array(lengths)
    n = len(lengths)
    print(f"\nDataset: {ds['name']}  ({n} examples)")
    print(f"mean={lengths.mean():.0f}  median={np.median(lengths):.0f}  max={lengths.max()}")
    print("percentiles: " + "  ".join(f"p{p}={int(np.percentile(lengths, p))}" for p in (50, 75, 90, 95, 99)))
    print()
    for t in sorted({512, 1024, 2048, args.limit}):
        over = int((lengths > t).sum())
        print(f"> {t:>5} tokens: {over:>5} / {n}  ({100 * over / n:.1f}%)  -> {n - over} would be kept")


if __name__ == "__main__":
    main()