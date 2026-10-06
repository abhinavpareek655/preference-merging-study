import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    
    messages = [
        {
            "role": "user",
            "content": "Explain what machine learning is in two sentences."
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
    )
    
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )
    
    print("\n===== MODEL RESPONSE =====\n")
    print(response)
    print("\n===== SUCCESS: Generation completed =====")

if __name__ == "__main__":
    main()
