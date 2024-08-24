# 导入所需的库
from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import BCEmbedding  
import os
from langchain.utilities import GoogleSerperAPIWrapper
import pprint


# 这个模型似乎无法这么从魔搭上下载，我们是自己导入的？直接pip的
# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/BCEmbeddingmodel', cache_dir='./')

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100

# 定义模型路径
model_path = './IEITYuan/Yuan2-2B-Mars-hf'

# 定义向量模型路径
embedding_model_path = './AI-ModelScope/BCEmbeddingmodel'

# 定义源大模型类
class Yuan2_LLM(LLM):
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List):#, context_web: str):
        if context:
            prompt = f'背景：{context}\n 问题：{question}\n 我现在正在申请计算机领域的博士，以上背景是关于学校和教授的学术信息，请根据以上信息回答我的问题'
        else:
            prompt = question

        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs, do_sample=False, max_length=2048)
        output = self.tokenizer.decode(outputs[0])

        return output.split("<sep>")[-1]

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"

# 新建大模型LLM 类
llm = LLM(model_path)

# 定义向量模型类
torch.cuda.empty_cache()

class EmbeddingModel:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
    
    def get_embeddings(self, sentences: List[str], batch_size: int = 8) -> np.ndarray:  
        all_embeddings = []  
        for i in range(0, len(sentences), batch_size):  
            batch = sentences[i:i + batch_size]  
            #print(batch)
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")  
            #print(inputs)
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}  
            with torch.no_grad():  
                outputs = self.model(**inputs_on_device, return_dict=True)  
            embeddings = outputs.last_hidden_state[:, 0]  
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  
            all_embeddings.append(embeddings.cpu().numpy())  
        return np.vstack(all_embeddings)  

    
# 新建 EmbeddingModel
embed_model = EmbeddingModel(embedding_model_path)


# 定义向量库索引类
class VectorStoreIndex:
    """
    class for VectorStoreIndex
    """

    def __init__(self, doecment_path: str, embed_model: EmbeddingModel) -> None:
        self.documents = []
        for line in open(doecment_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        self.embed_model = embed_model
        self.vectors = self.embed_model.get_embeddings(self.documents)

        print(f'Loading {len(self.documents)} documents for {doecment_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self, question: str, k: int = 2) -> List[str]:
        question_vector = self.embed_model.get_embeddings([question])[0]
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()
    '''
    def web_search(self, search_list):
        os.environ["SERPER_API_KEY"] = "88a8892a02409063f02a3bb97ac08b36fb213ae7"
        search = GoogleSerperAPIWrapper()
        search_result = ''
        for prof_name in search_list:
            search_item = prof_name + "research interest"
            search_result+= str(search.run(search_item)) + '\n'
            
        return search_result
            # results = search.results(search_item)
            # pprint.pp(results)
    '''

# 新建Index类
document_path = './test.txt'
index = VectorStoreIndex(document_path, embed_model)


# 页面布局
st.set_page_config(page_title="CS PhD申请助手", layout="wide")

# 标题和说明
st.title("CS PhD申请助手")
st.markdown("""
    你好，我是你的智能博士申请助手。请选择你的研究方向，系统将为你推荐合适的学校和导师，并提供基于大语言模型生成的个性化建议。
    """)

# 左侧栏用于选择研究方向
with st.sidebar:
    st.header("请选择研究方向")
    research_area = st.selectbox(
        "研究方向：",
        ("AI", "System", "Theory", "Interdisciplinary Areas")
    )

# 展示推荐学校及建议
if st.sidebar.button("提交研究方向"):
    # 根据选择的研究方向查询相关文档
    recommendations = index.query(research_area)
    
    # 使用聊天形式展示学校推荐
    with st.chat_message("assistant"):
        st.write(f"**根据你的研究方向：{research_area}，推荐以下学校和导师：**")
        for rec in recommendations:
            st.write(f"- {rec}")

    # 将检索到的推荐内容作为背景，生成更详细的推荐理由
    context = "\n".join(recommendations)
    
    # 使用LLM生成关于推荐的详细描述
    detailed_recommendation = llm.generate(
        question="请给出关于这些学校的详细推荐理由。并且介绍一下这些导师",
        context=context
    )
    
    # 使用聊天形式展示详细推荐理由
    with st.chat_message("assistant"):
        st.write(f"**详细推荐理由：**")
        st.write(detailed_recommendation)

# 保持对话上下文记录
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 用户输入问题
user_input = st.text_input("你可以在此输入问题：")

if user_input:
    # 记录用户的问题
    st.session_state["chat_history"].append(("user", user_input))

    # 显示用户的问题
    with st.chat_message("user"):
        st.write(user_input)

    # 根据用户输入生成回答
    response = llm.generate(question=user_input, context=context)

    # 记录LLM的回答
    st.session_state["chat_history"].append(("assistant", response))

    # 显示LLM的回答
    with st.chat_message("assistant"):
        st.write(response)

# 重新显示聊天记录
for role, message in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.write(message)