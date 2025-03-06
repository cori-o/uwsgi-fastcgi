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

class LLMModel():
    def __init__(self, config):
        self.config = config 

    def set_gpu(self, model):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"    
        model.to(self.device)
    
    def set_generation_config(self, max_tokens=500, temperature=0.9):
        self.gen_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }


class EmbModel():
    def __init__(self):
        pass 
    
    def set_gpu(self, model):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        model.to(self.device)

    def set_embbeding_config(self, batch_size=12, max_length=1024):
        self.emb_config = {
            "batch_size": batch_size, 
            "max_length": max_length 
        }
    
    def bge_embed_data(self, text):
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
        if isinstance(text, str):
            # encode result  => dense_vecs, lexical weights, colbert_vecs
            embeddings = model.encode(text, batch_size=self.emb_config['batch_size'], max_length=self.emb_config['max_length'])['dense_vecs']
        else:       
            embeddings = model.encode(list(text), batch_size=self.emb_config['batch_size'], max_length=self.emb_config['max_length'])['dense_vecs']  
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

class LLMOpenAI(LLMModel):
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


class LLMLlama(LLMModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "sh2orc/Llama-3.1-Korean-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.set_gpu(self.model)

    def set_generation_config(self, max_tokens=500, temperature=0.9):
        self.gen_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    def get_response(self, query, role="너는 금융권에서 일하고 있는 조수로, 회사 규정에 대해 알려주는 역할을 맡고 있어. 사용자 질문에 대해 간단 명료하게 답을 해줘."):
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": query}
            ]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            outputs = pipeline(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def set_prompt_template(self, query, context):
        self.rag_prompt_template = """
        다음 질문에 대해 주어진 정보를 참고해서 답을 해줘.
        주어진 정보: {context}
        --------------------------------
        질문: {query} 
        """
        return self.rag_prompt_template.format(query=query, context=context)


class LLMMistral(LLMModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(config['model_path'], config['model_type'], 'tokenizer'))
        self.model = AutoModelForCausalLM.from_pretrained(os.path.join(config['model_path'], config['model_type']),\
                                                    torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='cuda:0')
        self.set_gpu(self.model)

    def set_generation_config(self, temperature=0.8, do_sample=True, top_p=0.95, max_new_tokens=512): 
        self.gen_config = GenerationConfig(
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    def get_response(self, query):
        gened = self.model.generate(
            **self.tokenizer(
                query,
                return_tensors='pt',
                return_token_type_ids=False
            ).to(self.device),
            generation_config=self.gen_config,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # streamer=streamer,
        )
        
        result_str = self.tokenizer.decode(gened[0])
        start_tag = f"[/INST]"
        start_index = result_str.find(start_tag)
        print(result_str, end='\n\n')
        print(start_index)
        if start_index != -1:
            response = result_str[start_index + len(start_tag):].strip()
        else:
            return result_str
        
    def set_rag_prompt_template(self, query, context):
        self.prompt_template = (
            f"""
            ### <s> [INST]
            참고: 다음 질문에 대해 너의 금융 정보에 기반해서 답을 해줘. 참고할만한 정보는 다음과 같아. 
            {context}
            ### Question:
            {query}
            [/INST] """
        ) 