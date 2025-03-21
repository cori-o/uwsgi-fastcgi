from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig
from openai import OpenAI
from abc import ABC, abstractmethod
import numpy as np
import torch
import warnings
import torch
import os

# 특정 경고 메시지 무시
# warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class Model:
    def __init__(self, config):
        self.config = config 
        self.set_gpu()
       
    def set_gpu(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    def set_random_state(self, seed=42):
        self.random_state = seed


class EmbModel(Model):
    def __init__(self, config):
        super().__init__(config) 
    
    def set_embbeding_config(self, batch_size=12, max_length=1024):
        self.emb_config = {
            "batch_size": batch_size, 
            "max_length": max_length 
        }

    def set_emb_model(self, model_type='bge'):
        if model_type == 'bge':
            from FlagEmbedding import BGEM3FlagModel
            self.bge_emb = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
             
    def bge_embed_data(self, text):
        if isinstance(text, str):
            # encode result  => dense_vecs, lexical weights, colbert_vecs
            embeddings = self.bge_emb.encode(text, batch_size=self.emb_config['batch_size'], max_length=self.emb_config['max_length'])['dense_vecs']
        else:       
            embeddings = self.bge_emb.encode(list(text), batch_size=self.emb_config['batch_size'], max_length=self.emb_config['max_length'])['dense_vecs']  
        embeddings = list(map(np.float32, embeddings))
        return embeddings

    def calc_emb_similarity(self, emb1, emb2, metric='L2'):
        if metric == 'L2':   # Euclidean distance
            l2_distance = np.linalg.norm(emb1 - emb2)
            return l2_distance

    @abstractmethod
    def get_hf_encoder(self):
        pass

    @abstractmethod 
    def get_cohere_encoder(self, cohere_api):
        pass


class LLMOpenAI(Model):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI()

    def set_generation_config(self, max_tokens=500, temperature=0.9):
        self.gen_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    def get_response(self, query, role="너는 금융권에서 일하고 있는 조수로, 회사 규정에 대해 알려주는 역할을 맡고 있어. 사용자 질문에 대해 간단 명료하게 답을 해줘.", model='gpt-4'):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.gen_config['max_tokens'],
                temperature=self.gen_config['temperature'],
            )    
        except Exception as e:
            return f"Error: {str(e)}"
        return response.choices[0].message.content

    def set_prompt_template(self, query, context):
        self.rag_prompt_template = """
        다음 질문에 대해 주어진 정보를 참고해서 답을 해줘.
        주어진 정보: {context}
        --------------------------------
        질문: {query} 
        """
        return self.rag_prompt_template.format(query=query, context=context)

