# 安装依赖（在终端运行）
# pip install -q torch transformers peft datasets trl accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, AutoPeftModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# 加载数据集
dataset = load_dataset("hfl/alpaca_zh_51k", split="train")
print(dataset[:5])

# 加载模型和分词器
model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model.config.use_cache = False

# 配置 LoRA
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# 数据预处理
def preprocess_function(example):
    return {"text": f"### 指令: {example['instruction']}\n### 响应: {example['output']}"}


dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "output"])


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 配置训练参数
training_args = SFTConfig(
    output_dir="./qwen2-0.5b-peft",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    max_steps=500,
    warmup_steps=50,
)

# 初始化训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
)

# 开始训练
trainer.train()
trainer.model.save_pretrained("./qwen2-0.5b-peft-final")
tokenizer.save_pretrained("./qwen2-0.5b-peft-final")

# 推理测试
model = AutoPeftModelForCausalLM.from_pretrained(
    "./qwen2-0.5b-peft-final", quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./qwen2-0.5b-peft-final")
input_text = "### 指令: 1+1等于多少?\n### 响应:"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
