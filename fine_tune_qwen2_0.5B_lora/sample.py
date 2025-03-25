# 导入 PEFT（Parameter-Efficient Fine-Tuning）库中的 LoRA 相关工具
# LoraConfig: 用于配置 LoRA 的超参数
# get_peft_model: 将 LoRA 应用于基础模型
# PeftModel: 用于加载和操作经过 PEFT 微调的模型
from peft import LoraConfig, get_peft_model, PeftModel

# 导入 Transformers 库中的核心组件
# AutoModelForCausalLM: 自动加载因果语言模型（适用于生成任务）
# AutoTokenizer: 自动加载与模型匹配的分词器
# Trainer: 用于训练模型的工具类
# TrainingArguments: 配置训练参数的类
# DataCollatorForLanguageModeling: 数据整理器，用于语言建模任务的数据批处理
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 导入 ModelScope 的 MsDataset，用于加载和管理数据集
from modelscope.msdatasets import MsDataset

# 导入 PyTorch，用于张量操作和设备管理
import torch

# 检查设备
# torch.cuda.is_available(): 检查是否有可用的 GPU
# 如果有 GPU，则 device 设置为 "cuda"，否则为 "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # 输出当前使用的设备（GPU 或 CPU）

# 如果本地没有模型，则使用 modelscope 下载模型
# 如果 transformers 环境可行，也可用 transformers 库下载
# 下面的注释代码展示了如何从 ModelScope 下载模型（这里未启用）
# from modelscope import AutoModelForCausalLM, AutoTokenizer
# model_name = "Qwen/Qwen2-0.5B"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义本地模型路径
# 这里直接使用本地缓存的 Qwen2-0.5B 模型路径，避免在线下载
model_name = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen2-0.5B"

# 加载本地模型和 tokenizer
# AutoModelForCausalLM.from_pretrained: 从指定路径加载预训练因果语言模型
# AutoTokenizer.from_pretrained: 从指定路径加载对应的分词器
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"预训练模型加载成功")  # 确认模型和分词器加载成功

# 设置 padding token
# 检查 tokenizer 是否有 pad_token，如果没有，则将其设置为 eos_token（结束标记）
# pad_token 用于在批处理中对齐序列长度，eos_token 是句子的结束符号
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"开始加载数据集")  # 输出提示，表示数据集加载开始

# 加载并格式化数据集
# MsDataset.load: 从 ModelScope 加载名为 'llamafactory/alpaca_zh' 的数据集，subset_name 指定为 'default'
# to_hf_dataset(): 将加载的数据集转换为 Hugging Face 的 Dataset 格式（假设此方法可用）
dataset = MsDataset.load("llamafactory/alpaca_zh", subset_name="default").to_hf_dataset()

# 使用 map 函数对数据集进行格式化
# lambda example: 定义一个匿名函数，将每条数据格式化为 "Instruction: ... Input: ... Output: ..." 的字符串
# example['instruction']: 数据中的指令部分
# example['input']: 数据中的输入部分（如果存在）
# example['output']: 数据中的输出部分
dataset = dataset.map(
    lambda example: {
        "text": f"Instruction: {example['instruction']}"
        + (f"\nInput: {example['input']}" if example["input"] else "")  # 如果有 input，则添加，否则为空
        + f"\nOutput: {example['output']}"
    }
)

# 输出数据集的基本信息
print(f"原始数据集形状：{dataset.shape}")  # dataset.shape 显示数据集的行数和列数
print(dataset[:4])  # 输出前 4 条数据，便于检查格式化结果

# 分割训练集和验证集
len_data_set = len(dataset)  # 获取数据集的总长度
print(f"数据集长度 = {len_data_set}")  # 输出总数据条数

# 设置微调数据比例
my_fine_tune_data_len_rate = 0.01  # 只使用 1% 的数据进行微调
len_data_set = int(my_fine_tune_data_len_rate * len_data_set)  # 计算实际用于微调的数据量
print(f"实际参与微调的数据集长度 = {len_data_set}")  # 输出微调数据量

# 计算训练集和验证集的大小
train_size = int(0.8 * len_data_set)  # 训练集占微调数据的 80%
train_dataset = dataset.select(range(train_size))  # 选择前 train_size 条数据作为训练集
eval_dataset = dataset.select(range(train_size, len_data_set))  # 剩余部分作为验证集

# 输出分割结果
print(f"{train_size = }")  # 显示训练集大小
print(f"参与微调的训练集长度 = {len(train_dataset)}")  # 输出训练集实际长度
print(f"参与微调的验证集长度 = {len(eval_dataset)}")  # 输出验证集实际长度

# 对数据集进行 tokenization（分词和编码）
# map 函数对训练集进行分词
# tokenizer(...): 将文本转换为模型输入的 token IDs
# truncation=True: 如果文本超过 max_length，则截断
# padding="max_length": 不足 max_length 的序列用 pad_token 填充
# max_length=512: 设置最大序列长度为 512 个 token
# batched=True: 按批次处理，提高效率
# remove_columns=["text"]: 删除原始的 "text" 列，仅保留 tokenization 结果
train_dataset = train_dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
    remove_columns=["text"],
)

# 对验证集进行相同的 tokenization 处理
eval_dataset = eval_dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
    remove_columns=["text"],
)

print(train_dataset.shape)  # 输出 tokenization 后的训练集形状

# 配置 LoRA 参数
# LoraConfig: 定义 LoRA 的超参数
# r=8: LoRA 的秩（rank），控制低秩矩阵的大小，值越小参数越少
# lora_alpha=16: LoRA 的缩放因子，影响适配器权重的大小
# target_modules=["q_proj", "v_proj"]: 指定应用 LoRA 的模块（这里是注意力层的查询和值投影）
# lora_dropout=0.05: dropout 概率，防止过拟合
# bias="none": 不调整偏置参数
# task_type="CAUSAL_LM": 指定任务类型为因果语言建模
peft_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# 应用 LoRA 到模型
# get_peft_model: 将 LoRA 配置应用到预训练模型，只微调部分参数
model = get_peft_model(model, peft_config)

# 配置训练参数
# TrainingArguments: 定义训练过程中的超参数
# output_dir="./output": 训练输出目录，保存检查点和日志
# num_train_epochs=1: 训练 1 个 epoch
# per_device_train_batch_size=4: 每个设备（GPU/CPU）的训练批次大小
# per_device_eval_batch_size=4: 每个设备的验证批次大小
# warmup_steps=50: 学习率预热步数，逐步增加学习率
# weight_decay=0.01: 权重衰减，防止过拟合
# logging_dir="./logs": 日志保存目录
# logging_steps=100: 每 100 步记录一次日志
# evaluation_strategy="epoch": 每个 epoch 结束时评估模型
# save_strategy="epoch": 每个 epoch 结束时保存模型
# load_best_model_at_end=True: 训练结束时加载最佳模型（根据 metric_for_best_model）
# metric_for_best_model="loss": 用损失值评估最佳模型
# greater_is_better=False: 损失越小越好
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

# 配置数据整理器
# DataCollatorForLanguageModeling: 为语言建模任务准备批次数据
# tokenizer: 使用前面加载的分词器
# mlm=False: 不使用掩码语言建模（MLM），因为这是因果语言建模任务
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 初始化训练器
# Trainer: 封装训练过程，管理模型、数据和参数
# model: 使用 LoRA 微调的模型
# args: 训练参数
# train_dataset: 训练数据集
# eval_dataset: 验证数据集
# data_collator: 数据整理器
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
)

# 开始训练
# trainer.train(): 执行模型训练过程，包括前向传播、反向传播和参数更新
trainer.train()

# 保存模型和 tokenizer
# trainer.save_model: 保存微调后的模型到指定目录
# tokenizer.save_pretrained: 保存分词器到同一目录
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# 加载原始基础模型
# base_model_path: 原始预训练模型的路径
# AutoModelForCausalLM.from_pretrained: 加载未经微调的基础模型
base_model_path = model_name
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 加载 LoRA 权重并应用到基础模型
# lora_model_path: 微调后模型的保存路径
# PeftModel.from_pretrained: 将保存的 LoRA 权重加载并应用到基础模型
lora_model_path = "./final_model"
new_model = PeftModel.from_pretrained(base_model, lora_model_path)

# 加载 tokenizer
# AutoTokenizer.from_pretrained: 从微调模型路径加载分词器
new_tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# 将模型移到设备
# to(device): 将模型参数移动到指定设备（GPU 或 CPU）
new_model.to(device)
print("模型加载成功！")  # 确认模型加载完成

# 生成对话
# prompt: 输入的对话指令
prompt = "可再生能源的存在对环境有什么影响？"
# 将指令格式化为与训练数据一致的字符串
text = f"Instruction: {prompt}\nOutput: "

# 对输入进行分词并转换为张量
# new_tokenizer: 使用微调后的分词器
# return_tensors="pt": 返回 PyTorch 张量格式
# to(device): 将输入张量移到指定设备
model_inputs = new_tokenizer([text], return_tensors="pt").to(device)

# 生成文本
# generate: 使用模型生成后续 token
# max_new_tokens=200: 最多生成 200 个新 token
generated_ids = new_model.generate(model_inputs.input_ids, max_new_tokens=200)

# 提取生成的 token，去除输入部分
# 通过列表推导式，从生成的 token 中减去输入的 token，只保留新生成的输出部分
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

# 解码生成的文本
# batch_decode: 将 token IDs 解码为人类可读的文本
# skip_special_tokens=True: 跳过特殊标记（如 <eos>）
response = new_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 输出问题和生成的回答
print(f"\n\n问题：{prompt}, 回答: {response}\n\n")
