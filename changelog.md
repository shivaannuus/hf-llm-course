# Change Log

## venv

```bash
# pip install transformers
python3 -m pip install transformers
pip install torch

# 未用到
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

## note

* GPT-like （也被称作`自回归` Transformer 模型）
* BERT-like （也被称作`自动编码` Transformer 模型）
* BART/T5-like （也被称作`序列到序列`的 Transformer 模型
* 对其训练过的语言有统计学的理解(无监督), 再通过迁移学习（transfer learning）微调(Fine-tuning)(有监督)
* 监督方式: 预测下一个单词。这被称为因果语言建模(causal language modeling), 掩码语言建模(masked language modeling)，俗称完形填空
* Encoder-only 模型：适用于需要理解输入的任务，如句子分类和命名实体识别。
* Decoder-only 模型：适用于生成任务，如文本生成。
* Encoder-decoder 模型 或者 sequence-to-sequence 模型：适用于需要根据输入进行生成的任务，如翻译或摘要。

![alt text](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg)

* 编码器 “双向”（向前/向后）注意力，被称为自编码模型。
* “解码器”模型仅使用 Transformer 模型的解码器部分。在每个阶段，对于给定的单词，注意力层只能获取到句子中位于将要预测单词前面的单词。这些模型通常被称为自回归模型
