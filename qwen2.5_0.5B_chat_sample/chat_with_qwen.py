from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

device = "cpu" if not torch.cuda.is_available() else "cuda"  # 如果有GPU则使用GPU，否则使用CPU

model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 经过了指令微调的Qwen2模型，适合用于对话

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载配置
config = AutoConfig.from_pretrained(model_name)

# 你的问题
prompt = "1+1等于多少?"
# 生成对话
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
# 将对话转换为模型输入
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=20)
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
# 将生成的文本解码为字符串
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\n\n问题：{prompt}, 回答: {response}\n\n")
