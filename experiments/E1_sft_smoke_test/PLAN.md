# E1: 50-example LoRA SFT Smoke Test Plan

## Goal
Implement a smoke test for Supervised Fine-Tuning (SFT) using LoRA on Qwen/Qwen2.5-0.5B-Instruct with a tiny dataset (50 examples) to verify the training pipeline works before scaling up.

## Steps

### 1. Environment Setup
- Ensure we are in the Kaggle environment with necessary libraries (transformers, peft, torch, etc.)
- The script should be runnable directly in Kaggle (e.g., as a .py file or in a notebook)

### 2. Data Preparation
- Create a tiny dataset of 50 examples for instruction following.
- Option 1: Use a subset of a public dataset (e.g., Alpaca, Dolly) and take first 50 examples.
- Option 2: Generate synthetic examples if needed.
- Format: Each example should have an instruction and optionally input, and the expected output.

### 3. Model Loading
- Load the base model: Qwen/Qwen2.5-0.5B-Instruct
- Load the tokenizer.

### 4. LoRA Configuration
- Use PEFT library to prepare the model for LoRA training.
- Choose LoRA hyperparameters (r, alpha, dropout, target modules).

### 5. Training Setup
- Set up the trainer (e.g., using Hugging Face Trainer or custom loop).
- Use one GPU.
- Set training arguments for a very short run (e.g., max_steps=10, or num_train_epochs=0.1, per_device_train_batch_size=4, etc.)
- Use a small learning rate (e.g., 1e-4).

### 6. Training
- Run the training and monitor loss.
- Save the LoRA adapter weights after training.

### 7. Evaluation (Optional)
- Run a quick inference test to ensure the model can generate text.

### 8. Script Location
- Place the training script in `scripts/train_e1_smoke_test.py`

### 9. Documentation
- Create a README.md in `experiments/E1_sft_smoke_test/` explaining the experiment, how to run it, and expected results.

## Files to Create
- `scripts/train_e1_smoke_test.py`
- `experiments/E1_sft_smoke_test/README.md`
- `experiments/E1_sft_smoke_test/PLAN.md` (this file)

## Notes
- The script should be self-contained and runnable without modification in the Kaggle environment.
- We assume the Kaggle environment has the necessary libraries (as per requirements-kaggle.txt).
- If any additional libraries are needed, they should be installed via pip in the script (with checks to avoid re-installation).
