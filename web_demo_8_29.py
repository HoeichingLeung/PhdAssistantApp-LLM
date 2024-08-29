import torch  
import transformers  
import numpy as np  
import streamlit as st  
from typing import List  
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


# 定义模型路径  
model_path = './Task3/LLM-Research/Meta-Llama-3-8B-Instruct'  
embedding_model_path = './Task3/AI-ModelScope/BCEmbeddingmodel'  


# 文本切分类  
class TextSplitter:  
    def __init__(self, max_chunk_size=256, overlap=50):  
        self.max_chunk_size = max_chunk_size  
        self.overlap = overlap  

    def split_text(self, text: str):  
        words = text.split()  
        chunks = []  
        for i in range(0, len(words), self.max_chunk_size - self.overlap):  
            chunk = words[i:i + self.max_chunk_size]  
            chunks.append(' '.join(chunk))  
        return chunks  

# 定义源大模型类   
class LLM:  
    def __init__(self, model_path: str, device: str = "cpu") -> None:  
        print("Creating tokenizer...")  
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)  
        
        print("Creating model...")  
        self.model = transformers.AutoModelForCausalLM.from_pretrained(  
            model_path,  
            torch_dtype=torch.float32,  # 使用float32，避免在CPU上使用float16
            trust_remote_code=True
        ).to(device)  # 将模型加载到指定设备
        print(f'Loading Llama 3 model from {model_path}.')  
        self.device = device  # 保存设备信息

    def generate(self, question: str, context: str):  # 这里将 context 定义为字符串
        if context:  
            prompt = f'Background:{context}\nQuestion: {question}\nI am currently applying for a Ph.D. in computer science, and the above background provides academic information about schools and professors. Please use this information to answer my question.'  
        else:  
            prompt = question  

        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)  
        outputs = self.model.generate(  
            inputs,  
            do_sample=True,  
            max_new_tokens=256,  
            temperature=0.7,  
            top_p=0.9  
        )  
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        return output  

# 定义向量模型类  
class EmbeddingModel:  
    def __init__(self, model_name: str, device: str = 'cpu') -> None:  
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)  
        self.model = transformers.AutoModel.from_pretrained(model_name)  
        self.device = device  
        self.model.to(self.device)  
    
    def get_embeddings(self, sentences: List[str], batch_size: int = 2) -> np.ndarray:  # 将批次大小减少到2
        all_embeddings = []  
        for i in range(0, len(sentences), batch_size):  
            batch = sentences[i:i + batch_size]  
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")  
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}  
            with torch.no_grad():  
                outputs = self.model(**inputs_on_device, return_dict=True)  
            embeddings = outputs.last_hidden_state[:, 0]  
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  
            all_embeddings.append(embeddings.cpu().numpy())  
            
            # 手动释放不再需要的变量
            del inputs, inputs_on_device, outputs
            torch.cuda.empty_cache()  # 虽然在CPU上不需要，但可以用于释放未使用的内存
        return np.vstack(all_embeddings)  

# 定义向量库索引类  
class VectorStoreIndex:  
    def __init__(self, document_path: str, embed_model: EmbeddingModel, chunker: TextSplitter) -> None:  
        self.documents = []  
        self.chunks = []  
        self.chunker = chunker  
        
        # 加载文档并进行切分  
        for line in open(document_path, 'r', encoding='utf-8'):  
            line = line.strip()  
            self.documents.append(line)  
            # 文本切分  
            self.chunks.extend(self.chunker.split_text(line))  
        
        self.embed_model = embed_model  
        self.vectors = self.embed_model.get_embeddings(self.chunks)  

        print(f'Loaded {len(self.documents)} documents and {len(self.chunks)} chunks from {document_path}.')  

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:  
        dot_product = np.dot(vector1, vector2)  
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)  
        if not magnitude:  
            return 0  
        return dot_product / magnitude  

    def query(self, question: str, k: int = 3) -> List[str]:  
        question_vector = self.embed_model.get_embeddings([question])[0]  
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])  
        top_chunks = np.array(self.chunks)[result.argsort()[-k:][::-1]].tolist()  
        return top_chunks  


# 使用会话状态管理模型  
if 'llm' not in st.session_state:  
    st.session_state.llm = LLM(model_path)  
    print('LLM模型已初始化。')  

if 'embed_model' not in st.session_state:  
    st.session_state.embed_model = EmbeddingModel(embedding_model_path)  
    print('嵌入模型已初始化。')  

if 'index' not in st.session_state:  
    chunker = TextSplitter(max_chunk_size=256, overlap=50)  
    document_path = './test.txt'  
    st.session_state.index = VectorStoreIndex(document_path, st.session_state.embed_model, chunker)  
    print('索引已初始化。')  

# 页面布局  
st.set_page_config(page_title="CS PhD申请助手", layout="wide")  

# 标题和说明  
st.title("CS PhD申请助手")  
st.markdown("""  
    你好，我是你的智能博士申请助手。请选择你的研究方向，系统将为你推荐合适的学校和导师，并提供基于大语言模型生成的个性化建议。  
    """)

# 初始化聊天记录和上下文
if "chat_history" not in st.session_state:  
    st.session_state["chat_history"] = []  
if "context" not in st.session_state:
    st.session_state.context = ""  # **初始化context**

# 左侧栏用于选择研究方向  
with st.sidebar:  
    st.header("请选择研究方向")  
    research_area = st.selectbox(  
        "研究方向：",  
        ("AI", "System", "Theory", "Interdisciplinary Areas")  
    )  

# 展示推荐学校及建议  
if st.sidebar.button("提交研究方向"):  
    recommendations = st.session_state.index.query(research_area)  
    
    with st.chat_message("assistant"):  
        st.write(f"**根据你的研究方向：{research_area}，推荐以下学校和导师：**")  
        for rec in recommendations:  
            st.write(f"- {rec}")  

    # 更新context
    st.session_state.context = "\n".join(recommendations)  # **将context存储到st.session_state**

    detailed_recommendation = st.session_state.llm.generate(  
        question="请给出关于这些学校的详细推荐理由。并且介绍一下这些导师",  
        context=st.session_state.context  # **从st.session_state获取context**
    )  
    
    with st.chat_message("assistant"):  
        st.write(f"**详细推荐理由：**")  
        st.write(detailed_recommendation)  

# 用户输入问题
user_input = st.text_input("你可以在此输入问题：")  

if user_input:  
    st.session_state["chat_history"].append(("user", user_input))  

    with st.chat_message("user"):  
        st.write(user_input)  
    
    # 将用户输入附加到context中
    st.session_state.context += f"\n{user_input}"
    
    response = st.session_state.llm.generate(question=user_input, context=st.session_state.context)  # **使用更新后的context**
    st.session_state["chat_history"].append(("assistant", response))  

    with st.chat_message("assistant"):  
        st.write(response)  

# 重新显示聊天记录  
for role, message in st.session_state["chat_history"]:  
    with st.chat_message(role):  
        st.write(message)