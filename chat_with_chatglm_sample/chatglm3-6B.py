from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

model = model.eval()
response, history = model.chat(tokenizer, "what different between dog and cat?", history=[])
print(response)
