#!/usr/bin/env python3
"""
Show the fields of a Hugging Face dataset and a sample row.

Usage:
    python inspect_dataset.py open-r1/OpenR1-Math-220k
    python inspect_dataset.py openai/gsm8k --config main
    python inspect_dataset.py HuggingFaceH4/ultrachat_200k --split train_sft --n 2
"""
import argparse
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--config", default=None)
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=1, help="number of sample rows")
ap.add_argument("--chars", type=int, default=300, help="max chars shown per field")
args = ap.parse_args()

# streaming=True reads just the first rows, without downloading the whole dataset
ds = load_dataset(args.name, args.config, split=args.split, streaming=True)

for i, row in enumerate(ds):
    if i >= args.n:
        break
    print(f"\n=== {args.name} | split={args.split} | row {i} ===")
    for key, value in row.items():
        text = repr(value)
        if len(text) > args.chars:
            text = text[:args.chars] + f"... [{len(text)} chars total]"
        print(f"- {key} ({type(value).__name__}): {text}")