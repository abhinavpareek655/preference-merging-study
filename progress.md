# Progress: Preference Merging Study

**Last updated:** 2026-10-08

## Project Overview

**Title:** Does Alignment Survive the Merge?  
*A Controlled Study of Preference Retention and Low-Cost Post-Merge Repair in Small Language Models*

**Primary model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Primary preference method:** DPO  
**Secondary method:** GRPO (planned replication, not started)

### Research objective

The project studies whether preference alignment learned by a preference-tuned expert survives model merging, how retention changes with merge settings and expert composition, and whether inexpensive post-merge preference tuning can repair alignment loss.

The main planned metric is the **Preference Retention Ratio (PRR)**:

\[
PRR = \frac{S_{merge} - S_{base}}{S_{pref} - S_{base}}
\]

where `S_base` is the base-model preference score, `S_pref` is the preference-expert score, and `S_merge` is the merged-model score.

---

## Project Infrastructure

- **Code repository:** `abhinavpareek655/preference-merging-study`
- **Development branches:** `main`, `dev-abhinav`, `dev-vedant`
- **Hugging Face namespace:** `abhinav655`
- **Laptop role:** development, code, documentation, Git operations only
- **Training/evaluation/merging:** Kaggle GPU environment only
- **GPU used on Kaggle:** Tesla T4
- **Model storage:** Hugging Face
- **Code/config/documentation storage:** GitHub
- **Kaggle -> Hugging Face authentication:** `HF_TOKEN` through Kaggle Secrets

### Important workflow rule

All trained adapters and full model weights are stored on Hugging Face. Kaggle downloads the required models from Hugging Face for training, baking, evaluation, and merging. GitHub stores source code, configs, experiment metadata, plots, and documentation.

---

## Methodology Used So Far

### 1. Common base model

All experts use the same base checkpoint:

`Qwen/Qwen2.5-0.5B-Instruct`

This is required so that expert parameter changes are directly comparable during merging and delta analysis.

### 2. SFT experts

Three domain/instruction experts are trained with **Supervised Fine-Tuning (SFT)** using **LoRA**:

- `E_math`
- `E_code`
- `E_instr`

LoRA keeps the original model weights frozen and learns a low-rank update:

\[
\Delta W = BA
\]

Current LoRA configuration used for these experiments:

- rank `r = 8`
- `lora_alpha = 32`
- target modules: `q_proj`, `v_proj`
- dropout: `0.05`
- bias: `none`
- task type: causal language modeling
- seed: `42`

The project methodology requires LoRA adapters to be **baked into full model weights before merging**.

### 3. Preference expert

`E_pref` is trained with **Direct Preference Optimization (DPO)** using preference pairs containing:

- a prompt
- a chosen response
- a rejected response

DPO directly optimizes the relative preference for the chosen response over the rejected response, using the reference model as the comparison baseline.

Current DPO configuration:

- preference loss: sigmoid DPO loss
- `beta = 0.1`
- maximum sequence length: `512`
- LoRA `r = 8`
- `lora_alpha = 32`
- target modules: `q_proj`, `v_proj`
- dropout: `0.05`
- learning rate: `5e-5`
- batch size: `2`
- gradient accumulation: `4`
- fp16: enabled
- seed: `42`

---

## Datasets Used

### `E_math`

**Training dataset:** `open-r1/OpenR1-Math-220k`

Purpose: mathematical reasoning/domain specialization.

Current configured subset for the experiment: up to **2,000 examples**, with a **90/10 train-validation split**.

### `E_code`

**Training dataset:** `google-research-datasets/mbpp`

Purpose: code generation/programming specialization.

The current loader found **374 usable examples**:

- 337 training
- 37 validation

This is smaller than the configured 2,000-example cap and is recorded as an actual dataset limitation rather than being artificially padded.

### `E_instr`

**Training dataset:** `HuggingFaceH4/Bespoke-Stratos-17k`

Purpose: general instruction/reasoning specialization.

Current configured subset: **2,000 examples** with:

- 1,800 training
- 200 validation

The dataset contains reasoning-trace style content and special reasoning markers; this is documented as a dataset characteristic.

### `E_pref`

**Training dataset:** `trl-lib/ultrafeedback_binarized`

Purpose: preference alignment through DPO.

The project plan calls for a main preference-training regime of **2,000 preference pairs**, with smaller/larger data sizes reserved for later ablations.

A separate held-out preference evaluation set is required for E1 and later experiments, and must not overlap with the training pairs.

---

## Expert Status

| Expert | Method | Training Dataset | Status | Adapter HF | Baked Full Model HF |
|---|---|---|---|---|---|
| `E_math` | SFT + LoRA | `open-r1/OpenR1-Math-220k` | ✅ COMPLETED | `abhinav655/qwen25-math-expert` | `abhinav655/qwen25-math-expert-full` |
| `E_code` | SFT + LoRA | `google-research-datasets/mbpp` | ✅ COMPLETED | `abhinav655/qwen25-code-expert` | `abhinav655/qwen25-code-expert-full` |
| `E_instr` | SFT + LoRA | `HuggingFaceH4/Bespoke-Stratos-17k` | ✅ COMPLETED | `abhinav655/qwen25-instruction-expert` | `abhinav655/qwen25-instruction-expert-full` |
| `E_pref` | DPO + LoRA | `trl-lib/ultrafeedback_binarized` | ✅ COMPLETED | `abhinav655/qwen25-preference-expert` | `abhinav655/qwen25-preference-expert-full` |

**Note:** all four baked full-model repositories have been successfully uploaded to Hugging Face. 

---

## Completed Infrastructure and Fixes

### E0 — Base/infrastructure sanity check

✅ Completed.

Verified the core repository/Kaggle workflow and base-model loading before expert training.

### Dataset-loader fixes

✅ Completed.

- Fixed `E_instr` loader to handle its actual `messages`/`conversations` schema.
- Verified dry-run loading and 90/10 train-validation splits.
- Recorded the smaller usable MBPP set rather than fabricating a larger dataset.

### SFT expert training

✅ Completed for `E_math`, `E_code`, and `E_instr`.

All three LoRA adapters were trained and uploaded to Hugging Face.

### DPO pipeline

✅ Implemented and smoke-tested.

A DPO configuration issue involving `max_prompt_length` was fixed for the installed TRL version; the training pipeline now uses the supported `max_length` configuration.

### Kaggle environment fix

✅ Completed.

The original `torchao` installation was incompatible with the installed PEFT stack. The environment was updated to a compatible `torchao` version so DPO training could run successfully.

### LoRA baking

✅ Completed.

The four adapters were baked into full model weights and the full models were uploaded to Hugging Face.

The final baking summary showed:

- `E_math` ✅
- `E_code` ✅
- `E_instr` ✅
- `E_pref` ✅

A post-run summary typo caused the baking script to stop after the actual baking/upload work had completed; the models did not need to be rebaked solely because of that reporting typo.

---

## E1 — Sanity Evaluation (Current Stage)

**Status:** ⏳ NEXT

The purpose of E1 is to verify that each expert improves the intended capability relative to the common base model before any merging experiments are trusted.

### Planned evaluation mapping

| Expert | Training | Intended E1 Evaluation |
|---|---|---|
| `E_math` | OpenR1-Math-220k | GSM8K |
| `E_code` | MBPP | MBPP test with genuine execution-based pass@1 |
| `E_instr` | Bespoke-Stratos-17k | IFEval |
| `E_pref` | UltraFeedback | Held-out UltraFeedback preference accuracy and DPO margin |

### Required E1 comparisons

Each specialist must be compared **against the same base model**, using separate plots rather than one combined multi-model plot.

Planned figures:

- `plots/e1_base_vs_math.png`
- `plots/e1_base_vs_code.png`
- `plots/e1_base_vs_instruction.png`
- `plots/e1_base_vs_preference_accuracy.png`
- `plots/e1_base_vs_preference_margin.png`

### Important E1 evaluation corrections

The earlier evaluation implementation was found to be unsuitable for final research reporting because it had several methodological problems. These must be fixed before E1 results are considered valid:

1. Preference evaluation must use a **true held-out split** that cannot overlap with the exact E_pref training examples.
2. Qwen prompts/conversations must be formatted with `tokenizer.apply_chat_template(...)` rather than manually constructed chat strings.
3. MBPP evaluation must execute generated code against the official tests for genuine pass@1 rather than checking whether output merely looks like Python.
4. IFEval should use a real IFEval-based evaluation rather than a small hand-written keyword proxy.
5. Expert-load failures must be treated as errors; silently substituting the base model would invalidate the experiment.
6. Final plots should be base-vs-specialist comparisons as defined above.

---

## Planned Delta and Merge Methodology

After E1 is validated:

### Expert delta extraction

For each baked expert:

\[
\Delta_i = W_i - W_{base}
\]

The same Qwen base checkpoint is used for every expert so that these deltas are directly comparable.

### Merge methods

Primary planned merge methods:

1. **Linear / weighted averaging**
2. **Task Arithmetic**
3. **TIES**
4. **DARE-TIES**

The first merge experiment should remain intentionally simple: a two-model merge between `E_pref` and `E_math` using the linear merge.

---

## Experiment Roadmap

### E1 — Expert sanity check

Verify each expert improves its intended capability relative to base.

**Status:** ⏳ Current next stage

### E2 — Preference retention curve

Merge `E_pref` with another expert and sweep the merge coefficient:

\[
\alpha \in \{0.0, 0.1, 0.2, \ldots, 1.0\}
\]

Measure preference retention and capability changes.

**First target:** `E_pref + E_math` with linear merging.

**Status:** ⏳ Future

### E3 — Consortium size

Study the effect of merging more experts on preference retention and capability retention.

**Status:** ⏳ Future

### E4 — Workflow order

Compare:

- Align → merge
- Merge → align
- Preference-anchor merging

**Status:** ⏳ Future

### E5 — Low-cost post-merge repair

Test whether a small amount of post-merge preference tuning restores lost alignment efficiently.

**Status:** ⏳ Future

### E6 — Preference data size

Compare smaller/larger preference-training budgets, including the planned 500 / 2,000 / 8,000 style ablation where feasible.

**Status:** ⏳ Future

### E7 — Merge algorithm ablation

Compare linear merging, task arithmetic, TIES, and DARE-TIES.

**Status:** ⏳ Future

### E8 — Conflict predictor

Analyze whether parameter-level signals such as delta cosine similarity, sign conflict, and delta norm predict preference retention.

**Status:** ⏳ Future

### E9 — GRPO replication

Replicate the main preference-retention experiment with GRPO as a secondary method.

**Status:** ⏳ Secondary future work

### E10 — 1.5B replication

Repeat the core experiment with a larger model if compute/time permits.

**Status:** ⏳ Secondary future work

### E11 — Seed robustness

Repeat key experiments across multiple seeds and report mean/std or confidence intervals.

**Status:** ⏳ Future

### E12 — Evaluator robustness

Check whether conclusions are stable under reasonable evaluator changes.

**Status:** ⏳ Future

---

## Metrics Planned

### Preference metrics

- Preference accuracy
- DPO margin / chosen-vs-rejected log-probability difference
- Preference Retention Ratio (PRR)

### Capability metrics

- GSM8K accuracy for math
- MBPP execution-based pass@1 for code
- IFEval score for instruction following

### Merge/conflict diagnostics

- Delta cosine similarity
- Sign conflict rate
- Delta norm
- Parameter-space conflict statistics

### Repair efficiency

- Preference recovery per training step
- Preference recovery per training example
- Capability retained during repair

---

## Reproducibility Rules

- Keep the base model fixed.
- Use the same tokenizer across experts.
- Use seed `42` for the main run unless a specific experiment changes the seed.
- Record exact dataset IDs and subset sizes.
- Record training hyperparameters.
- Keep expert adapters and baked full models separately identifiable.
- Use a held-out preference set that never overlaps with E_pref training data.
- Do not silently fall back to another model when a checkpoint fails to load.
- Save experiment metadata, scores, plots, and configurations with each experiment.
- Keep the main experiments small and reproducible before attempting GRPO or 1.5B replication.

---

## Current Project Status Summary

- **E0 infrastructure/base:** ✅ COMPLETED
- **E_math:** ✅ TRAINED + ✅ ADAPTER UPLOADED + ✅ BAKED FULL MODEL UPLOADED
- **E_code:** ✅ TRAINED + ✅ ADAPTER UPLOADED + ✅ BAKED FULL MODEL UPLOADED
- **E_instr:** ✅ TRAINED + ✅ ADAPTER UPLOADED + ✅ BAKED FULL MODEL UPLOADED
- **E_pref:** ✅ DPO PIPELINE + ✅ SMOKE TEST + ✅ ADAPTER UPLOADED + ✅ BAKED FULL MODEL UPLOADED; ⚠️ FINAL 2k/100-step RESEARCH RUN STILL NEEDS VERIFICATION
- **LoRA baking:** ✅ COMPLETED
- **E1 sanity evaluation:** ⏳ CURRENT NEXT
- **Expert delta extraction:** ⏳ AFTER E1
- **Merge implementation:** ⏳ AFTER E1
- **E2 retention curves:** ⏳ FUTURE
- **E3–E8:** ⏳ FUTURE
- **E9 GRPO:** ⏳ SECONDARY FUTURE
- **E10 1.5B replication:** ⏳ SECONDARY FUTURE
- **E11 seeds:** ⏳ FUTURE
- **E12 evaluator robustness:** ⏳ FUTURE
- **Paper/statistics/figures:** ⏳ FUTURE

---

## Immediate Next Action

**Do not rebake the models again unless a checkpoint is missing.**

The immediate next action is:

> **Verify the final E_pref checkpoint, then run the corrected E1 sanity evaluation on the base model and all four baked experts.**

After E1 is validated, proceed to expert delta extraction and the first simple two-model linear merge (`E_pref + E_math`, sweeping `alpha = 0.0 ... 1.0`).

---

## Key Hugging Face Repositories

### LoRA adapters

- `abhinav655/qwen25-math-expert`
- `abhinav655/qwen25-code-expert`
- `abhinav655/qwen25-instruction-expert`
- `abhinav655/qwen25-preference-expert`

### Baked full models

- `abhinav655/qwen25-math-expert-full`
- `abhinav655/qwen25-code-expert-full`
- `abhinav655/qwen25-instruction-expert-full`
- `abhinav655/qwen25-preference-expert-full`

---

## One-Line Project State

**All four expert pipelines have been built, trained/uploaded, and baked to full models; the project is now at the corrected E1 sanity-evaluation stage, with only the final E_pref training-checkpoint verification remaining before the merge experiments begin.**