"""
https://huggingface.co/learn/llm-course/zh-CN/chapter1


目前 可用的一些pipeline 有：

eature-extraction （获取文本的向量表示）
fill-mask （完形填空）
ner （命名实体识别）
question-answering （问答）
sentiment-analysis （情感分析）
summarization （提取摘要）
text-generation （文本生成）
translation （翻译）
zero-shot-classification （零样本分类）

"""

from transformers import pipeline
import torch 

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_capability(0))
print(torch.cuda.is_bf16_supported())
print(torch.cuda.is_mps_available())
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
print(torch.backends.cuda.is_built())
print(torch.backends.cuda.is_available())
print(torch.__version__)
print("let's begin")

classifier = pipeline("sentiment-analysis")
classifier("This is a great day!")