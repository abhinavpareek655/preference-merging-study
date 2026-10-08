# Progress: Preference Merging Study

**Last updated**: 2026-10-06

## Project Overview
Title: Does Alignment Survive the Merge?
A Controlled Study of Preference Retention and Low-Cost Post-Merge Repair in Small Language Models

Primary model: Qwen/Qwen2.5-0.5B-Instruct
Primary method: DPO (with GRPO as secondary replication)

## Current Expert Status

| Expert | Method | Dataset | Status | Hugging Face |
|--------|--------|---------|--------|--------------|
| E_math | SFT + LoRA | open-r1/OpenR1-Math-220k | COMPLETED | abhinav655/qwen25-math-expert |
| E_code | SFT + LoRA | google-research-datasets/mbpp | COMPLETED | abhinav655/qwen25-code-expert |
| E_instr | SFT + LoRA | HuggingFaceH4/Bespoke-Stratos-17k | COMPLETED | abhinav655/qwen25-instruction-expert |
| E_pref | DPO + LoRA | trl-lib/ultrafeedback_binarized | SMOKE TEST COMPLETE; FINAL RUN TO VERIFY | abhinav655/qwen25-preference-expert |

## Immediate Next Steps
1. Verify that the final E_pref 2000-example DPO run exists.
2. Verify/reload all four LoRA adapters.
3. Bake each LoRA adapter into full model weights.
4. Compute each expert delta relative to the common Qwen base.
5. Implement the merge infrastructure.
6. Evaluate B0/B1/B2 sanity conditions.
7. Start E2 with the two-model linear merge: E_pref + E_math, alpha = 0.0 ... 1.0
8. Add TIES and DARE-TIES.
9. Calculate PRR.
10. Move to consortium size and workflow-order experiments.

## Current Project Status Summary

- **E0 infrastructure/base**: ✅ COMPLETED
- **E_math**: ✅ TRAINED, ✅ UPLOADED TO HUGGING FACE
- **E_code**: ✅ TRAINED, ✅ UPLOADED TO HUGGING FACE
- **E_instr**: ✅ TRAINED, ✅ UPLOADED TO HUGGING FACE
- **E_pref**: ✅ DPO PIPELINE CREATED, ✅ SMOKE TEST COMPLETED, ✅ ADAPTER UPLOADED, ⚠️ FINAL 2k/100-step RESEARCH RUN MUST BE VERIFIED
- **LoRA baking**: ⏳ NEXT
- **Expert delta extraction**: ⏳ NEXT
- **Merge implementation**: ⏳ NEXT
- **E1 sanity evaluation**: ⏳ NEXT
- **E2 retention curves**: ⏳ FUTURE
- **E3–E8**: ⏳ FUTURE
- **E9 GRPO**: ⏳ SECONDARY FUTURE
- **E10 1.5B replication**: ⏳ SECONDARY FUTURE
- **E11 seeds**: ⏳ FUTURE
- **E12 evaluator robustness**: ⏳ FUTURE
- **Paper/statistics/figures**: ⏳ FUTURE

**Current next action**: Verify/finalize E_pref, then bake all four LoRA adapters into full model weights.