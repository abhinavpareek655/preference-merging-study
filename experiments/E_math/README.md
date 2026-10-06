# E_math: SFT Expert for Mathematical Reasoning

This experiment trains a supervised fine-tuning (SFT) expert on mathematical reasoning using the OpenR1-Math-220k dataset.

## Goal
To create an expert model that excels at solving mathematical problems by fine-tuning Qwen/Qwen2.5-0.5B-Instruct on a subset of OpenR1-Math-220k.

## Dataset
- Source: open-r1/OpenR1-Math-220k
- Subset size: 2,000 examples
- Split: 90% training, 10% validation
- Format: Each example consists of a problem (instruction) and its answer (output)

## Training
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- LoRA configuration: r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
- Training script: `scripts/train_sft_expert.py --config configs/expert_math.yaml`
- Output directory: `outputs/E_math`

## Hyperparameters
See `configs/expert_math.yaml` for full training hyperparameters.

## Expected Outcome
The expert should demonstrate improved performance on mathematical reasoning tasks compared to the base model.

## Files Generated
- `outputs/E_math`: Directory containing the LoRA adapter weights and training state.
