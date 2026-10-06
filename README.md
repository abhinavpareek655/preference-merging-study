# Does Alignment Survive the Merge?

## Preference Retention and Low-Cost Repair in Small Language Models

This project studies whether preference alignment learned through DPO survives
when independently fine-tuned language models are merged.

## Research Question

Does preference tuning survive model merging, and how much post-merge
preference data is required to recover lost alignment?

## Primary Model

0.5B-scale instruction model.

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
