from dotenv import load_dotenv
from pymilvus import Collection
from .milvus import MilvusEnvManager, DataMilVus, MilvusMeta
from .query import OpenAIQT, RulebookQR
from .llm import EmbModel, LLMOpenAI
import json
import os

class EnvManager():
    def __init__(self, args):
        self.args = args
        load_dotenv()
        self.ip_addr = os.getenv('ip_addr')
        self.cohere_api = os.getenv('COHERE_API_KEY')   

    def set_config(self):
        with open(os.path.join(self.args['config_path'], self.args['db_config'])) as f:
            self.db_config = json.load(f)
        with open(os.path.join(self.args['config_path'], self.args['llm_config'])) as f:
            self.llm_config = json.load(f)
            
    def set_vectordb(self):
        self.db_config['ip_addr'] = self.ip_addr
        self.milvus_db = MilvusEnvManager(self.db_config)
        self.milvus_db.set_env()
        data_milvus = DataMilVus(self.db_config)
        meta_milvus = MilvusMeta()
        meta_milvus.set_rulebook_map()
        rulebook_eng_to_kor = meta_milvus.rulebook_eng_to_kor

        self.collection = Collection(self.args['collection_name'])
        self.collection.load()

        self.milvus_db.get_partition_info(self.collection)
        self.partition_list = [rulebook_eng_to_kor[p_name] for p_name in self.milvus_db.partition_names if not p_name.startswith('_')]
        return data_milvus

    def set_llm(self):
        emb_model = EmbModel()   # Embedding Model: bge-m3
        emb_model.set_embbeding_config()
        response_model = LLMOpenAI(self.llm_config)   # Response Generator
        response_model.set_generation_config()
        return emb_model, response_model 

    def set_query_transformer(self, use_query_transform):
        if use_query_transform == False: 
            self.query_transformer = None 
        else:
            self.query_transformer = OpenAIQT(self.llm_config)   # Query Translator
            self.query_transformer.set_generation_config()

    def set_query_router(self, use_query_routing):
        if use_query_routing == False: 
            self.query_router = None
            self.route_layer = None 
        else:
            self.query_router = RulebookQR()   # Query Router
            self.query_router.create_prompt_injection_utterances()
            self.query_router.create_rulebook_utterances()
            prompt_injection_route = self.query_router.create_route('prompt_injection', self.query_router.prompt_injection_utterances)
            rulebook_route = self.query_router.create_route('rulebook_check', self.query_router.prompt_rulebook_check_utterances)
            route_encoder = self.query_router.get_cohere_encoder(self.cohere_api)
            self.route_layer = self.query_router.create_route_layer(route_encoder, [prompt_injection_route, rulebook_route])