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

在 https://huggingface.co/models 找模型

"""

from transformers import pipeline
import torch 

def print_cuda_info():
    print("torch.__version__=", torch.__version__)
    print("torch.cuda.is_available()=", torch.cuda.is_available())
    print("torch.cuda.get_device_name(0)=", torch.cuda.get_device_name(0))
    print("torch.cuda.current_device()=", torch.cuda.current_device())
    print("torch.cuda.device_count()=", torch.cuda.device_count())
    print("torch.cuda.get_device_capability(0)=", torch.cuda.get_device_capability(0))
    print("torch.cuda.is_bf16_supported()=", torch.cuda.is_bf16_supported())
    print("torch.backends.cuda.is_built()=", torch.backends.cuda.is_built())
    print("")

def print_each(txt, infoseq):
    print( txt, ":\r\n")
    for info in infoseq:
        print(info)

    print("")

if __name__ == "__main__":
    print_cuda_info()
    print("let's begin!!")

    # 使用pipeline进行情感分析
    classifier = pipeline("sentiment-analysis")
    sentiment_ret_str = classifier(["This is a great day!", "the quick brown fox jumps over the lazy dog."])
    print_each("sentiment_ret_str", sentiment_ret_str)    

    # 使用pipeline进行零样本分类
    zero_classifier = pipeline("zero-shot-classification")
    zero_ret_str = zero_classifier("this is a course about transformers",
                                candidate_labels=["education", "politics", "business"])    
    print_each("zero_ret_str", zero_ret_str)

    # 使用特定模型进行文本生成
    text_generator = pipeline("text-generation", model="distilgpt2")
    text_gen_ret_str = text_generator("Hello, my name is Houy and I am a", max_length=50, num_return_sequences=3)
    text_gen_ret_str2 = text_generator("Cwm fjord bank glyphs vext quiz.", max_length=50, num_return_sequences=3)
    print_each("text_gen_ret_str", text_gen_ret_str)
    print_each("text_gen_ret_str2", text_gen_ret_str2)
