# E_code: SFT Expert for Code Generation

This experiment trains a supervised fine-tuning (SFT) expert on code generation using the MBPP dataset.

## Goal
To create an expert model that excels at generating Python code by fine-tuning Qwen/Qwen2.5-0.5B-Instruct on the MBPP dataset.

## Dataset
- Source: google-research-datasets/mbpp
- Subset size: 2,000 training examples
- Split: 90% training, 10% validation
- Format: Each example consists of a problem description (instruction) and the corresponding code (output)

## Training
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- LoRA configuration: r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
- Training script: `scripts/train_sft_expert.py --config configs/expert_code.yaml`
- Output directory: `outputs/E_code`

## Hyperparameters
See `configs/expert_code.yaml` for full training hyperparameters.

## Expected Outcome
The expert should demonstrate improved performance on code generation tasks compared to the base model.

## Files Generated
- `outputs/E_code`: Directory containing the LoRA adapter weights and training state.
