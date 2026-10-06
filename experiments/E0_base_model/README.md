# E0 — Base Model Sanity Check

## Model

Qwen/Qwen2.5-0.5B-Instruct

## Purpose

Verify that the base model can be downloaded, loaded and used for
text generation.

## Status

In progress.

## Expected result

The model should successfully generate a response to a basic instruction.
## Related Experts

After verifying the base model, we train three SFT experts:
- E_math: Supervised fine-tuning on OpenR1-Math-220k (2,000 examples)
- E_code: Supervised fine-tuning on MBPP (2,000 training examples)
- E_instruction: Supervised fine-tuning on Bespoke-Stratos-17k (2,000 examples)

Each expert uses LoRA (r=8, q_proj/v_proj) and is saved under `outputs/E_math`, `outputs/E_code`, and `outputs/E_instr` respectively.
