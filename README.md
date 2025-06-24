# ai_model_samples

本仓库包含一些与 AI 模型相关的 demo。

每个模型为单独的文件目录，相关模型demo 的介绍参考各目录下的 README.md 文件。

每个目录下都有对应 requirements.txt 文件，按照 requirements.txt 中给出的各 python 包的版本进行安装。

推荐安装方法：
```shell
pip install -r requirements.txt
```

# 更多学习教程
> 
> 原创 AI 学习系列教程，点击这里：[AI小白到AI大神的天梯之路](https://www.yuque.com/yuqueyonghupftxbc/ai100/snfie4aka5fn5ykg)
>
> 加入《小而精的 AI 学习圈》知识星球，一起学习 AI ： [2025我们星球见](https://www.yuque.com/yuqueyonghupftxbc/ai100/aiy5vb3bu2id5agp)

---
# 项目目录

## 视觉项目
- 利用 resnet50 + 宠物数据集进行重训练：[点这里](https://github.com/dongdongcan/ai_model_samples/tree/main/resnet50_train_oxford_iiit_pet)
- 利用 resnet50 + 宠物数据集进行微调：[点这里](https://github.com/dongdongcan/ai_model_samples/tree/main/resnet50_fine_tune_oxford_iiit_pet)
- 利用 resnet50 进行推理，识别图片:[点这里](./resnet50_inference_sample/)
- 利用 yolo 完成图像的目标检测：[点这里](./yolo_detection_sample/)
- 利用 yolo_V3 完成图像的目标检测：[点这里](./yolo_v3_detection_sample/)
- 多种模型和方法完成人脸识别：[点这里](./face_detect/)
- 多种模型和方法完成手写数字识别：[点这里](./mnist_sample/)


## 大模型项目
- 利用 qwen2-0.5B 进行对话演示:[点这里](./chat_with_qwen2_0.5B_sample/)
- 利用 qwen2.5-0.5B 进行对话演示:[点这里](./chat_with_qwen2.5_0.5B_sample/)
- 利用 qwen2.5-3B-instruct 进行对话演示:[点这里](./chat_with_qwen2.5_3B_instruct/)
- 利用 llama 进行对话演示:[点这里](./chat_with_llama_sample/)
- 利用 chatglm 进行对话演示:[点这里](./chat_with_chatglm_sample/)
- 利用 vllm 部署 qwen2-0.5B:[点这里](./vllm_qwen2_0.5B_deploy/)
- OpenAI 项目，在线部署/前端页面/API部署:[点这里](./openai/)

## 微调项目
- 利用 peft/lora 微调 qwen2-0.5B:[点这里](./fine_tune_qwen2_0.5B_lora/)

---
## 联系我
- 微信号：ddcsggcs
- 公众号：[董董灿是个攻城狮](https://mp.weixin.qq.com/s/9sdmLFcNWnASmzpYNIhQKQ?token=273250015&lang=zh_CN)
- 邮箱：dongdongcan2024@163.com。



