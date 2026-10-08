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
  - Running a simple generation test
  - Confirming the model can be saved and reloaded

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login, whoami
import getpass

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

def login_to_hf():
    """Login to Hugging Face if not already logged in."""
    try:
        whoami()
        print("Already logged in to Hugging Face")
    except Exception:
        print("Please enter your Hugging Face token:")
        token = getpass.getpass()
        login(token)
        print("Logged in to Hugging Face")

def main():
    # Login to Hugging Face
    login_to_hf()

    # Initialize HF API
    api = HfApi()

    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    manifest = []

    for expert in EXPERTS:
        print(f"\nProcessing {expert}...")
        adapter_repo = ADAPTER_REPOS[expert]
        baked_repo = BAKED_REPOS[expert]

        try:
            # Load the PEFT model from Hugging Face
            print(f"  Loading adapter from {adapter_repo}")
            model = PeftModel.from_pretrained(base_model, adapter_repo)

            # Merge and unload
            print(f"  Merging LoRA weights...")
            merged_model = model.merge_and_unload()

            # Save merged model and tokenizer temporarily for validation
            temp_dir = f"/tmp/{expert}"
            os.makedirs(temp_dir, exist_ok=True)

            print(f"  Saving baked model to temporary directory {temp_dir}")
            merged_model.save_pretrained(temp_dir)
            base_tokenizer.save_pretrained(temp_dir)

            # Validation
            print(f"  Validating baked model...")
            validation_success = validate_baked_model(temp_dir, base_tokenizer)

            if validation_success:
                # Upload to Hugging Face
                print(f"  Uploading baked model to {baked_repo}")
                try:
                    # Check if repo exists, if not create it
                    try:
                        api.repo_info(baked_repo)
                        print(f"    Repository {baked_repo} already exists")
                    except Exception:
                        print(f"    Creating repository {baked_repo}")
                        api.create_repo(repo_id=baked_repo, exist_ok=True)

                    # Upload the baked model
                    api.upload_folder(
                        folder_path=temp_dir,
                        repo_id=baked_repo,
                        repo_type="model",
                    )
                    print(f"    Successfully uploaded to {baked_repo}")

                    baking_status = "success"
                except Exception as e:
                    print(f"    Error uploading to HF: {e}")
                    baking_status = "upload_failed"
            else:
                baking_status = "validation_failed"

            # Record manifest entry
            manifest_entry = {
                "expert": expert,
                "base_model": BASE_MODEL_NAME,
                "adapter_source": adapter_repo,
                "baked_output_repo": baked_repo,
                "seed": 42 if expert == "E_pref" else "unknown",  # seed known for E_pref from config
                "baking_status": baking_status,
                "validation_passed": validation_success if 'validation_success' in locals() else False,
            }
            manifest.append(manifest_entry)

            if baking_status == "success":
                print(f"  {expert}: Baking, validation, and upload succeeded.")
            else:
                print(f"  {expert}: Baking completed with status: {baking_status}")

        except Exception as e:
            print(f"  {expert}: Error during baking: {e}")
            manifest.append({
                "expert": expert,
                "base_model": BASE_MODEL_NAME,
                "adapter_source": adapter_repo,
                "baked_output_repo": baked_repo,
                "seed": "unknown",
                "baking_status": "error",
                "error": str(e),
            })

    # Write manifest locally
    manifest_path = "./manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Summary
    print("\n=== Baking Summary ===")
    for entry in manifest:
        print(f"{entry['expert']}: {entry['baking_status']}")

def validate_baked_model(model_path, tokenizer):
    """
    Validate the baked model by:
      1. Reloading with AutoModelForCausalLM
      2. Running a simple generation test
      3. Checking that the model can be saved and reloaded
    """
    try:
        # Reload the baked model
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Reload the tokenizer from the baked model's directory
        reloaded_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Simple generation test
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt").to(reloaded_model.device)

        # Generate
        with torch.no_grad():
            outputs = reloaded_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    Generation test: '{prompt}' -> '{generated_text}'")

        # Test saving and reloading again (to a temporary directory)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "test_reload")
            reloaded_model.save_pretrained(test_path)
            reloaded_tokenizer.save_pretrained(test_path)
            # Reload again to ensure it works
            AutoModelForCausalLM.from_pretrained(test_path)
            AutoTokenizer.from_pretrained(test_path)

        return True
    except Exception as e:
        print(f"    Validation error: {e}")
        return False

if __name__ == "__main__":
    main()