#!/usr/bin/env python3
"""
E1: 50-example LoRA SFT Smoke Test for Qwen/Qwen2.5-0.5B-Instruct

This script runs a supervised fine-tuning smoke test using LoRA on a tiny dataset
of 50 examples. It is designed to run quickly on a single GPU in Kaggle.
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# For reproducibility
torch.manual_seed(42)

def main():
    # Model name
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # For causal LM, we need to set the padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 to save memory
    )
    model.to(device)

    # Prepare model for kbit training if using quantization (optional)
    # We are not using quantization in this smoke test, but we can prepare for it if needed.
    # model = prepare_model_for_kbit_training(model)  # Requires importing prepare_model_for_kbit_training

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Common target modules for Qwen
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model for LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create a tiny dataset of 50 examples
    print("Creating dataset...")
    # We'll create a simple instruction-following dataset
    examples = []
    for i in range(50):
        # Simple examples: instruction to repeat a word or phrase
        word = f"word{i}"
        examples.append({
            "instruction": f"Repeat the word '{word}'",
            "input": "",  # No input
            "output": word,
        })

    # Format examples into chat format
    def format_example(example):
        # Format as a conversation
        messages = [
            {"role": "user", "content": example["instruction"] + (" " + example["input"] if example["input"] else "")},
            {"role": "assistant", "content": example["output"]},
        ]
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # We are training on the full sequence
        )
        return {"text": text}

    formatted_examples = [format_example(ex) for ex in examples]
    dataset = Dataset.from_list(formatted_examples)

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./e1_smoke_test_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        max_steps=10,  # Very short training run
        learning_rate=1e-4,
        fp16=True,  # Use float16 precision
        logging_steps=1,
        save_strategy="no",  # We don't need to save checkpoints during training
        report_to="none",  # Disable reporting to integrations for simplicity
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    print("Saving LoRA adapter...")
    trainer.save_model("./e1_smoke_test_lora_adapter")

    # Optional: Test generation with the fine-tuned model
    print("Testing generation...")
    model.eval()
    test_instruction = "Repeat the word 'test'"
    messages = [
        {"role": "user", "content": test_instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_instruction}")
    print(f"Response: {response}")

    print("Smoke test completed successfully!")

if __name__ == "__main__":
    main()