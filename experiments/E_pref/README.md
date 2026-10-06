# E_pref — Preference Expert

## Objective

Train a preference-aligned expert from the fixed base model:

Qwen/Qwen2.5-0.5B-Instruct

using Direct Preference Optimization (DPO).

## Model

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Parameter-efficient method: LoRA
- LoRA rank: 8
- LoRA alpha: 32
- Target modules: `q_proj`, `v_proj`

## Preference Dataset

- Dataset: `trl-lib/ultrafeedback_binarized`
- Training split: `train`
- Maximum examples used: 2000
- Train/validation split: 90/10
- Random seed: 42

Each training example contains:

- prompt
- chosen response
- rejected response

The dataset loader converts the conversational dataset into an
explicit preference format for TRL's DPOTrainer.

## DPO Configuration

- Beta: 0.1
- Loss: sigmoid DPO loss
- Maximum sequence length: 512
- Maximum prompt length: 256
- Learning rate: 5e-5
- Maximum training steps: 100
- Training batch size per device: 2
- Gradient accumulation: 4
- FP16: enabled

## Training Command

```bash
python scripts/train_dpo_expert.py \
    --config configs/expert_preference.yaml
```

## Dry Run

```bash
python scripts/train_dpo_expert.py \
    --config configs/expert_preference.yaml \
    --dry-run
```

## Smoke Test

```bash
python scripts/train_dpo_expert.py \
    --config configs/expert_preference.yaml \
    --smoke-test
```

## Output

The trained LoRA adapter is saved to:

```text
outputs/E_pref/
```

The adapter will later be uploaded to:

```text
abhinav655/qwen25-preference-expert
```

## Role in the Research

E_pref is the preference-aligned expert used to study whether
preference tuning survives model merging.

The main research metric is Preference Retention Ratio (PRR).