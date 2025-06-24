import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(torch.randint(0, 10000, (1,)).item())

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()


def chat_with_model(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").cuda()

    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    response = chat_with_model(user_input)
    print("GPT-2: " + response)
