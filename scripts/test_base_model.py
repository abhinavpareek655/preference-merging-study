from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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

    inputs = tokenizer(text, return_tensors="pt")

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


if __name__ == "__main__":
    main()