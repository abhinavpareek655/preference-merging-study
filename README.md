# Does Alignment Survive the Merge?

## Preference Retention and Low-Cost Repair in Small Language Models

This project studies whether preference alignment learned through DPO survives
when independently fine-tuned language models are merged.

## Research Question

Does preference tuning survive model merging, and how much post-merge
preference data is required to recover lost alignment?

## Primary Model

Qwen/Qwen2.5-0.5B-Instruct

## Experts

- Math expert
- Code expert
- Instruction expert
- Preference expert

## Primary Method

DPO

## Merge Methods

- Linear
- TIES
- DARE-TIES

## Main Metrics

- Preference Retention Ratio (PRR)
- Preference accuracy
- DPO margin
- Domain capability retention
- Parameter conflict

## Team

- Abhinav
- Vedant

## Status

Project initialization.

## Setup and Resources

- **GitHub**: Source code and configuration
- **Kaggle**: GPU training and experiments
- **Hugging Face**: Model storage, adapters, and checkpoints
- **Primary Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Note**: No model weights should be committed to GitHub
