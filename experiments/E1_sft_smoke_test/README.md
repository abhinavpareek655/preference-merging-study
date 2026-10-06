# E1: 50-example LoRA SFT Smoke Test

This experiment runs a supervised fine-tuning (SFT) smoke test using LoRA on the Qwen/Qwen2.5-0.5B-Instruct model with a tiny dataset of 50 examples.

## Goal
To verify that the training pipeline works correctly before scaling up to larger datasets and more extensive training.

## Contents
- `scripts/train_e1_smoke_test.py`: The training script that performs the smoke test.
- This README.

## How to Run
The script is designed to be runnable directly in a Kaggle environment.

### In Kaggle Notebook
You can run the script by executing:
```bash
!python /path/to/preference-merging-study/scripts/train_e1_smoke_test.py
```
Alternatively, you can copy the script into a notebook cell and run it.

### In Kaggle Scripts
Create a new script and run:
```bash
python /path/to/preference-merging-study/scripts/train_e1_smoke_test.py
```

## Expected Output
The script will:
1. Load the Qwen/Qwen2.5-0.5B-Instruct model and tokenizer.
2. Create a synthetic dataset of 50 instruction-following examples.
3. Apply LoRA configuration (r=8, alpha=32, target modules: q_proj, v_proj).
4. Train for 10 steps (very short run) with a batch size of 4 and gradient accumulation of 2.
5. Save the LoRA adapter to `./e1_smoke_test_lora_adapter`.
6. Test generation with a sample prompt to verify the model still works.

## Notes
- The script uses float16 precision to save memory.
- It assumes a GPU is available (as typical in Kaggle). If no GPU is found, it will fall back to CPU (but training will be very slow).
- The dataset is intentionally simple: each example asks the model to repeat a word. This is to ensure the task is easy to learn quickly.
- After training, you should see a reduction in training loss and the model should be able to generate the expected output for the test prompt.

## Troubleshooting
- If you encounter out-of-memory errors, try reducing the batch size or gradient accumulation steps.
- Ensure you have the required libraries installed (transformers, peft, torch, datasets). They are available in the Kaggle environment via the provided requirements.

## Files Generated
- `./e1_smoke_test_output`: Training output directory (checkpoints, etc.) - note: we set save_strategy="no" so no checkpoints are saved during training.
- `./e1_smoke_test_lora_adapter`: Directory containing the saved LoRA adapter weights.

## Related Experts

The smoke test (E1) validates the training pipeline for the following SFT experts:
- E_math: Supervised fine-tuning on OpenR1-Math-220k (2,000 examples)
- E_code: Supervised fine-tuning on MBPP (2,000 training examples)
- E_instruction: Supervised fine-tuning on Bespoke-Stratos-17k (2,000 examples)

Each expert uses the same LoRA configuration as E1 (r=8, q_proj/v_proj) and is trained with the reusable script `scripts/train_sft_expert.py`.
