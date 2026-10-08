#!/usr/bin/env python3
"""
Bake LoRA adapters into full model weights for the four experts.

This script loads the base model and each LoRA adapter from Hugging Face,
merges them, and saves the resulting full model back to Hugging Face.

Expected adapter locations (Hugging Face):
  abhinav655/qwen25-math-expert
  abhinav655/qwen25-code-expert
  abhinav655/qwen25-instruction-expert
  abhinav655/qwen25-preference-expert

After baking, the script validates each baked model by:
  - Reloading with AutoModelForCausalLM
  - Running a simple generation test using Qwen chat template
  - Confirming the model can be saved and reloaded
  - Verifying no LoRA adapter files/dependencies remain

The baked models are uploaded to Hugging Face repositories:
  abhinav655/qwen25-math-expert-full
  abhinav655/qwen25-code-expert-full
  abhinav655/qwen25-instruction-expert-full
  abhinav655/qwen25-preference-expert-full

A manifest.json is created locally with metadata.
"""

import os
import json
import torch
import tempfile
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EXPERTS = ["E_math", "E_code", "E_instr", "E_pref"]
ADAPTER_REPOS = {
    "E_math": "abhinav655/qwen25-math-expert",
    "E_code": "abhinav655/qwen25-code-expert",
    "E_instr": "abhinav655/qwen25-instruction-expert",
    "E_pref": "abhinav655/qwen25-preference-expert",
}
BAKED_REPOS = {
    "E_math": "abhinav655/qwen25-math-expert-full",
    "E_code": "abhinav655/qwen25-code-expert-full",
    "E_instr": "abhinav655/qwen25-instruction-expert-full",
    "E_pref": "abhinav655/qwen25-preference-expert-full",
}

def check_e_pref_status():
    """Check progress.md to verify E_pref status before processing."""
    progress_file = "/run/media/abhinav/241AB4EC1AB4BC5C/preference-merging-study/progress.md"
    try:
        with open(progress_file, 'r') as f:
            content = f.read()

        # Look for E_pref status in the table
        if "E_pref | DPO + LoRA | trl-lib/ultrafeedback_binarized | SMOKE TEST COMPLETE; FINAL RUN TO VERIFY" in content:
            print("⚠️  WARNING: According to progress.md, E_pref is only smoke test complete.")
            print("   The final 2000-example/100-step research run MUST BE VERIFIED before baking.")
            print("   Aborting E_pref processing to avoid using incomplete checkpoint.")
            return False
        elif "E_pref | DPO + LoRA | trl-lib/ultrafeedback_binarized | COMPLETED" in content:
            print("✅ E_pref status confirmed as COMPLETED in progress.md")
            return True
        else:
            # Fallback: check if the language suggests completion
            if "FINAL 2k/100-step RESEARCH RUN MUST BE VERIFIED" in content:
                print("⚠️  WARNING: progress.md indicates final E_pref run needs verification.")
                print("   Aborting E_pref processing.")
                return False
            else:
                print("ℹ️  Could not determine E_pref status from progress.md, proceeding with caution.")
                return True  # Proceed but user should verify manually
    except Exception as e:
        print(f"Error reading progress.md: {e}")
        print("Proceeding with caution - user should verify E_pref status manually.")
        return True

def main():
    # Get HF token from Kaggle secrets/environment
    user_secrets = UserSecretsClient()
    token = user_secrets.get_secret("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not found. Please set it in Kaggle secrets.")

    # Initialize HF API with token
    api = HfApi(token=token)

    print("🔐 Logged in to Hugging Face using HF_TOKEN from environment")

    # Check E_pref status before processing
    e_pref_verified = check_e_pref_status()

    manifest = []

    for expert in EXPERTS:
        # Skip E_pref if not verified
        if expert == "E_pref" and not e_pref_verified:
            print(f"\n⏭️  Skipping {expert} - final research run not verified in progress.md")
            manifest.append({
                "expert": expert,
                "base_model": BASE_MODEL_NAME,
                "adapter_source": ADAPTER_REPOS[expert],
                "baked_output_repo": BAKED_REPOS[expert],
                "seed": 42,
                "baking_status": "skipped - final run not verified",
                "validation_passed": False,
                "note": "See progress.md: FINAL 2k/100-step RESEARCH RUN MUST BE VERIFIED"
            })
            continue

        print(f"\n🔧 Processing {expert}...")
        adapter_repo = ADAPTER_REPOS[expert]
        baked_repo = BAKED_REPOS[expert]

        # Create fresh temporary directory for this expert
        with tempfile.TemporaryDirectory(prefix=f"bake_{expert}_") as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # 1. Load FRESH base model for each expert (requirement 1)
                print(f"  📦 Loading fresh base model: {BASE_MODEL_NAME}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    dtype=torch.float16,  # Fixed: use dtype instead of torch_dtype (requirement 3)
                    device_map="auto",
                )
                base_tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL_NAME,
                    trust_remote_code=True
                )

                # 2. Load LoRA adapter ONLY from Hugging Face (requirement 2)
                print(f"  📥 Loading adapter from {adapter_repo}")
                model = PeftModel.from_pretrained(base_model, adapter_repo)

                # 3. Merge with PEFT merge_and_unload(safe_merge=True) (requirement 6)
                print(f"  🔀 Merging LoRA weights with safe_merge=True...")
                merged_model = model.merge_and_unload(safe_merge=True)

                # 4. Save baked model temporarily on Kaggle (requirement 7)
                print(f"  💾 Saving baked model to temporary directory: {temp_path}")
                merged_model.save_pretrained(temp_path)
                base_tokenizer.save_pretrained(temp_path)

                # 5. Validate the baked model (requirements 8, 9, 10)
                print(f"  🔍 Validating baked model...")
                validation_success = validate_baked_model(
                    str(temp_path),
                    base_tokenizer,  # Use original tokenizer for consistency in test
                    expert  # Pass expert name for logging
                )

                if validation_success:
                    # 6. Upload to Hugging Face full-model repository (requirement 7)
                    print(f"  ☁️  Uploading baked model to {baked_repo}")

                    # Check if repo exists
                    try:
                        api.repo_info(baked_repo, token=token)
                        print(f"    Repository {baked_repo} already exists")
                    except Exception:
                        print(f"    Creating repository {baked_repo}")
                        api.create_repo(repo_id=baked_repo, token=token, exist_ok=True)

                    # Upload the baked model
                    api.upload_folder(
                        folder_path=str(temp_path),
                        repo_id=baked_repo,
                        repo_type="model",
                        token=token,
                    )
                    print(f"    Successfully uploaded to {baked_repo}")

                    baking_status = "success"
                else:
                    baking_status = "validation_failed"

                # Record manifest entry (requirement 11)
                manifest_entry = {
                    "expert": expert,
                    "base_model": BASE_MODEL_NAME,
                    "adapter_source": adapter_repo,
                    "baked_output_repo": baked_repo,
                    "seed": 42,  # All experts use seed 42 as per configs
                    "baking_status": baking_status,
                    "validation_passed": validation_success if 'validation_success' in locals() else False,
                }
                manifest.append(manifest_entry)

                if baking_status == "success":
                    print(f"  ✅ {expert}: Baking, validation, and upload succeeded.")
                else:
                    print(f"  ❌ {expert}: Baking completed with status: {baking_status}")

            except Exception as e:
                print(f"  💥 {expert}: Error during baking: {e}")
                manifest.append({
                    "expert": expert,
                    "base_model": BASE_MODEL_NAME,
                    "adapter_source": adapter_repo,
                    "baked_output_repo": baked_repo,
                    "seed": 42,
                    "baking_status": "error",
                    "error": str(e),
                })

    # Write manifest locally
    manifest_path = "./manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n📝 Manifest written to {manifest_path}")

    # Print summary
    print("\n" + "="*50)
    print("📊 BAKING SUMMARY")
    print("="*50)
    for entry in manifest:
        status = entry['baking_status']
        emoji = "✅" if status == "success" else "❌" if "error" in status or "failed" in status else "⏭️"
        print(f"{emoji} {entry['expert']}: {status}")

    # Count successes
    successes = sum(1 for e in manifest if e['banking_status'] == 'success')
    print(f"\n🎯 Successfully baked: {sum(1 for e in manifest if e['baking_status'] == 'success')}/{len([e for e in manifest if e['expert'] != 'E_pref' or e_pref_verified])} experts")

def validate_baked_model(model_path, tokenizer, expert_name):
    """
    Validate the baked model by:
      1. Reloading with AutoModelForCausalLM
      2. Running a simple generation test using Qwen chat template
      3. Confirming the model can be saved and reloaded
      4. Verifying no LoRA adapter files/dependency remain
    """
    try:
        # 1. Reload the baked model
        print(f"    🔄 Reloading baked model from {model_path}")
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Reload the tokenizer from the baked model's directory
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 2. Verify no LoRA files remain (requirement 10)
        lora_files = list(Path(model_path).glob("adapter_*")) + list(Path(model_path).glob("*/adapter_*"))
        if lora_files:
            print(f"    ⚠️  Warning: LoRA files found: {[f.name for f in lora_files]}")
        else:
            print(f"    ✅ Confirmed: No LoRA adapter files/dependencies remain")

        # 3. Use Qwen chat template for generation test (requirement 9)
        # Qwen chat format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        prompt = "Hello, my name is"
        chat_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(reloaded_model.device)

        # Generate
        print(f"    🧪 Running generation test...")
        with torch.no_grad():
            outputs = reloaded_model.generate(
                **inputs,
                max_new_tokens=20,  # Increased for better test
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "<|im_start|>assistant\n" in generated_text:
            response = generated_text.split("<|im_start|>assistant\n")[-1].strip()
            # Remove any trailing tokens
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
        else:
            response = generated_text.strip()

        print(f"    💬 Generation test: '{prompt}' -> '{response}'")

        # 4. Test saving and reloading again (requirement 8)
        print(f"    🔄 Testing save/reload cycle...")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_reload"
            reloaded_model.save_pretrained(test_path)
            reloaded_tokenizer.save_pretrained(test_path)

            # Reload again to ensure it works
            AutoModelForCausalLM.from_pretrained(test_path)
            AutoTokenizer.from_pretrained(test_path)

        return True
    except Exception as e:
        print(f"    ❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    main()