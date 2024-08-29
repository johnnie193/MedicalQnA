from typing import List
from langchain.embeddings.base import Embeddings
from mindnlp.sentence import SentenceTransformer


class EmbeddingsFunAdapter(Embeddings):
    def __init__(self, embed_model, mirror='huggingface'):
        self.embed_model = embed_model
        self.embedding_model = SentenceTransformer(model_name_or_path=self.embed_model, mirror=mirror)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode_texts(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embedding_model.encode_texts([text])
        return embeddings[0]

import json
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

class LineByLineJSONLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                # 提取并格式化 JSON 数据
                content = (
                    f"Name: {json_obj.get('name', '')}\n"
                    f"Description: {json_obj.get('desc', '')}\n"
                    f"Category: {', '.join(json_obj.get('category', []))}\n"
                    f"Prevent: {json_obj.get('prevent', '')}\n"
                    f"Cause: {json_obj.get('cause', '')}\n"
                    f"Symptom: {', '.join(json_obj.get('symptom', []))}\n"
                    f"Get Probability: {json_obj.get('get_prob', '')}\n"
                    f"Get Way: {json_obj.get('get_way', '')}\n"
                    f"Accompany: {', '.join(json_obj.get('acompany', []))}\n"
                    f"Cure Department: {', '.join(json_obj.get('cure_department', []))}\n"
                    f"Common Drug: {', '.join(json_obj.get('common_drug', []))}\n"
                    f"Recommend Drug: {', '.join(json_obj.get('recommand_drug', []))}\n"
                    f"Easy Get: {', '.join(json_obj.get('easy_get', []))}\n"
                    f"Medical Insurance status: {json_obj.get('yibao_status', '')}\n"
                    f"Cure Way: {', '.join(json_obj.get('cure_way', []))}\n"
                    f"Cure Last Time: {json_obj.get('cure_lasttime', '')}\n"
                    f"Cured Probability: {json_obj.get('cured_prob', '')}\n"
                    f"Cost Money: {json_obj.get('cost_money', '')}\n"
                    f"Check: {', '.join(json_obj.get('check', []))}\n"
                    f"Recommend Drug: {', '.join(json_obj.get('recommand_drug', []))}\n"
                    f"Drug Detail: {', '.join(json_obj.get('drug_detail', []))}\n"
                    f"Do Eat: {', '.join(json_obj.get('do_eat', []))}\n"
                    f"Don't eat: {', '.join(json_obj.get('not_eat', []))}\n"
                    f"Recommend Eat: {', '.join(json_obj.get('recommand_eat', []))}\n"
                )
                
                # 创建文档对象
                documents.append(Document(page_content=content, metadata={'id': json_obj.get('_id', '')}))
        return documents
# parser = argparse.ArgumentParser()
# parser.add_argument('filepath', type=str)
# args = parser.parse_args()
# 使用自定义加载器
loader = LineByLineJSONLoader("medical.json")
documents = loader.load()
# 文本切分
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = []
for doc in documents:
    split_docs = splitter.split_text(doc.page_content)
    for split_doc in split_docs:
        split_documents.append(Document(page_content=split_doc, metadata=doc.metadata))

import argparse
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st

import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

def load_knowledge_base(split_documents):
    texts = [doc.page_content for doc in documents]
    embeddings = EmbeddingsFunAdapter("AI-ModelScope/m3e-base", mirror='modelscope')
    faiss = FAISS.from_texts(texts, embeddings)
    return faiss


def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, ms_dtype=mindspore.float16)
    model.set_train(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer, model

# 返回对应的knowledge
def retrieve_knowledge(faiss, query):
    docs = faiss.similarity_search(query)
    return docs[0].page_content

# 将knowlege和query叠加生成input，调用模型chat返回response
def generate_answer(tokenizer, model, query, knowledge):
    prompt = f'作为一名顶级医疗助手，你的主要职责是提供精准、专业且富有同理心的医疗建议。你具备全面的医疗知识，能够实时更新并反映最新的医学进展。你的工作方式始终以患者为中心，确保提供最高标准的护理和清晰的沟通。技能：专家诊断与建议，深入分析患者提供的信息，结合最新医学知识进行精准诊断。提出科学合理的治疗建议，确保患者理解并信任你的建议。规则：在任何情况下都不允许破坏角色设定。不说废话，不编造事实。\n\n已知内容:{knowledge}\n\n问题:\n\n{query}'
    #这一步要再formulate下加个prompt
    messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": "你是谁?"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="ms"
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    response = model.generate(
            input_ids,
            max_new_tokens=20,
            eos_token_id=terminators,
            do_sample=False,
            # do_sample=True,
            # temperature=0.6,
            # top_p=0.9,
        )
    answer = response[0]
    return answer

def rag_pipeline(faiss, tokenizer, model, query, use_rag):
    if use_rag:
        knowledge = retrieve_knowledge(faiss, query)
        answer = generate_answer(tokenizer, model, query, knowledge)
        return answer, knowledge
    else:
        answer = generate_answer(tokenizer, model, query, "")
        return answer, ""
    
model_path = "01ai/Yi-6B-Chat"
st.title("AI医疗诊断大师")
faiss = load_knowledge_base(split_documents)
tokenizer, model = load_model_and_tokenizer(model_path)

if 'answer' not in st.session_state:
    st.session_state['answer'] = ""

if 'knowledge' not in st.session_state:
    st.session_state['knowledge'] = ""

with st.form(key='chat_form', clear_on_submit=True):
    query = st.text_input("输入您的问题：", "")
    use_rag = st.checkbox("启用检索增强生成 (RAG)", value=True)
    submitted = st.form_submit_button("发送")

if submitted and query:
    answer, knowledge = rag_pipeline(faiss, tokenizer, model, query, use_rag)
    st.session_state['answer'] = answer
    st.session_state['knowledge'] = knowledge

elif submitted:
    st.warning("请输入一个医疗诊断相关的问题。")

with st.subheader("用户输入"):
    st.text_area("User Query", query, height=50)

    # 左右布局
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("回答")
    st.text_area("Assistant Answer", st.session_state['answer'], height=300)

with right_column:
    st.subheader("知识检索")
    st.text_area("Knowledge", st.session_state['knowledge'], height=300)

    # 重新运行以刷新对话历史
if submitted:
    st.rerun()