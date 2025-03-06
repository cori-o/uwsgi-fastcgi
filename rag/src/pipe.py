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


class ChatUser():
    def __init__(self, vectordb=None, emb_model=None, response_model=None, query_transformer=None, query_router=None, route_layer=None, logger=None):
        self.vectordb = vectordb
        self.emb_model = emb_model
        self.response_model = response_model 
        self.query_transformer = query_transformer 
        self.query_router = query_router 
        self.route_layer = route_layer
        self.logger = logger  
    
    def transform_query(self, query):
        rewrite_q = self.query_transformer.rewrite_query(query)
        self.query_transformer.get_query_status('Query Rewriting', query, rewrite_q)
        stepback_q = self.query_transformer.stepback_query(query)
        self.query_transformer.get_query_status('Stepback Prompting', rewrite_q, stepback_q)
        return stepback_q

    def route_query(self, query):
        self.logger.info("Step 2. Query를 Routing합니다.")
        routed_query = self.query_router.semantic_layer(self.route_layer, query)
        self.logger.info(f"Routed query: {routed_query}")

        # 부적절한 발화가 입력된 경우 처리
        if self.route_layer(query).name == 'prompt_injection':
            response = self.response_model.get_response(routed_query)
            return response, True  # 대화를 종료
        return routed_query, False

    def retrieve_data(self, query, collection, output_fields='text'):
        cleansed_text = self.vectordb.cleanse_text(query)
        query_emb = self.emb_model.bge_embed_data(cleansed_text)
        self.vectordb.set_search_params(query_emb, output_fields=output_fields)   
        search_params = self.vectordb.search_params
        self.search_result = self.vectordb.search_data(collection, search_params)

    def postprocess_data(self, threshold):
        id_list = []; dist_list = []
        '''
        if reranking is needed
        '''
        retreived_txt = self.vectordb.decode_search_result(self.search_result)
        id_list, dist_list = self.vectordb.get_distance(self.search_result)
        threshold_txt = self.vectordb.check_l2_threshold(retreived_txt, threshold, dist_list[0])
        return threshold_txt

    def continue_conv(self):
        continue_conv = input('계속 대화하시겠습니까 ? (y/n): ')
        if continue_conv.lower() == 'y':
            flag = True 
            print(f'대화를 계속합니다.')
        else:
            flag = False
            print(f'대화를 종료합니다.')
        return flag 