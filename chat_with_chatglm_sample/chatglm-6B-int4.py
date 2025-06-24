from transformers import AutoTokenizer, AutoModel
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "Hello", history=[])
print(response)
