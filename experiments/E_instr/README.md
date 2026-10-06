# E_instruction: SFT Expert for Instruction Following

This experiment trains a supervised fine-tuning (SFT) expert on general instruction following using the Bespoke-Stratos-17k dataset.

## Goal
To create an expert model that excels at following diverse instructions by fine-tuning Qwen/Qwen2.5-0.5B-Instruct on a subset of Bespoke-Stratos-17k.

## Dataset
- Source: HuggingFaceH4/Bespoke-Stratos-17k
- Subset size: 2,000 examples
- Split: 90% training, 10% validation
- Format: Each example consists of a prompt (instruction) and a response (output)

## Training
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- LoRA configuration: r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
- Training script: `scripts/train_sft_expert.py --config configs/expert_instruction.yaml`
- Output directory: `outputs/E_instr`

## Hyperparameters
See `configs/expert_instruction.yaml` for full training hyperparameters.

## Expected Outcome
The expert should demonstrate improved performance on instruction following tasks compared to the base model.

## Files Generated
- `outputs/E_instr`: Directory containing the LoRA adapter weights and training state.
