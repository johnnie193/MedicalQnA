# 基于ChatGLM4的医疗问答机器人

# 模型简介

> 本方案旨在针对医疗问答场景，提出一个全面的技术解决方案。首先，我们对特定问答任务的数据集进行详细的数据预处理，以确保数据的质量和一致性。在此基础上，采用LoRA（Low-Rank Adaptation）技术对ChatGLM4模型进行微调，以提高模型在医疗领域的专用性和响应准确度。接下来，我们引入了检索增强生成（RAG）技术，利用与疾病相关的高质量数据集，并采用M3E作为嵌入模型，对知识库中的内容和用户输入进行相关性检索。这一过程确保了在用户提问的同时，模型能够及时获取所需的相关知识，提升问答的精准度和实用性。为了进一步优化模型的回答质量，我们设计了一个精细的Prompt框架，通过明确的指令和角色设定，指导模型扮演一个专业的医疗助手角色。这种设计使得模型的回答更加符合医疗领域的标准与期望，从而为用户提供更可靠、更专业的医疗建议。

## 数据集

> Huatuo-26M 数据集
https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets

> Disease-KG 数据集
https://github.com/honeyandme/RAGQnASystem.git

## 环境要求

pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
!wget https://mindspore-demo.obs.cn-north-4.myhuaweicloud.com/model/ZhipuAI/glm-4-9b-chat.zip
!mkdir -p ./.mindnlp/model/ZhipuAI
!unzip -d ./.mindnlp/model/ZhipuAI/ glm-4-9b-chat.zip

## 快速入门

python run.py
