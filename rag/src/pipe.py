from .milvus import MilvusEnvManager, DataMilVus, MilvusMeta
from .data_p import DataProcessor
from .models import EmbModel
from pymilvus import Collection
import json
import os

class EnvManager():
    def __init__(self, args):
        self.args = args
        self.set_config()
        self.cohere_api = os.getenv('COHERE_API_KEY')   
        self.db_config['ip_addr'] = self.args['ip_addr']
        
    def set_config(self):
        with open(os.path.join(self.args['config_path'], self.args['db_config'])) as f:
            self.db_config = json.load(f)
        with open(os.path.join(self.args['config_path'], self.args['llm_config'])) as f:
            self.llm_config = json.load(f)
    
    def set_processors(self):
        self.data_p = DataProcessor()
        
    def set_vectordb(self):
        self.milvus_db = MilvusEnvManager(self.db_config)
        data_milvus = DataMilVus(self.db_config)
        meta_milvus = MilvusMeta()
        return data_milvus, meta_milvus 

    def set_emb_model(self):
        emb_model = EmbModel(self.llm_config)
        emb_model.set_emb_model(model_type='bge')
        emb_model.set_embbeding_config()
        return emb_model         


class InteractManager:
    def __init__(self, data_p=None, vectorenv=None, vectordb=None, emb_model=None, response_model=None, logger=None):
        '''
        vectordb = MilvusData - insert data, set search params, search data 
        '''
        self.data_p = data_p
        self.vectorenv = vectorenv
        self.vectordb = vectordb 
        self.emb_model = emb_model 
        self.response_model = response_model 
        self.logger = logger 
    
    def create_domain(self, domain_name):
        '''
        domain = collection
        '''
        data_doc_id = self.vectorenv.create_field_schema('doc_id', dtype='VARCHAR', is_primary=True, max_length=1024)
        data_passage_id = self.vectorenv.create_field_schema('passage_id', dtype='INT64')
        data_domain = self.vectorenv.create_field_schema('domain', dtype='VARCHAR', max_length=32)
        data_title = self.vectorenv.create_field_schema('title', dtype='VARCHAR', max_length=128)
        data_text = self.vectorenv.create_field_schema('text', dtype='VARCHAR', max_length=512)   # 500B (500글자 단위로 문서 분할)
        data_text_emb = self.vectorenv.create_field_schema('text_emb', dtype='FLOAT_VECTOR', dim=1024)
        data_info = self.vectorenv.create_field_schema('info', dtype='JSON')
        data_tags = self.vectorenv.create_field_schema('tags', dtype='JSON')
        schema_field_list = [data_doc_id, data_passage_id, data_domain, data_title, data_text, data_text_emb, data_info, data_tags]

        schema = self.vectorenv.create_schema(schema_field_list, 'schema for fai-rag, using fastcgi')
        collection = self.vectorenv.create_collection(domain_name, schema, shards_num=2)
        self.vectorenv.create_index(collection, field_name='text_emb')   # doc_id 필드에 index 생성 

    def delete_data(self, domain, doc_id):
        hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
        data_to_delete = f"doc_id == {hashed_doc_id}"  
        self.vectordb.delete_data(filter=data_to_delete, collection_name=domain)

    def insert_data(self, domain, doc_id, title, text, info, tags):
        hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
        chunked_texts = self.data_p.chunk_text(text)
        for chunk in chunked_texts:
            chunk, passage_id = chunk[0], chunk[1]
            chunk_emb = self.emb_model.bge_embed_data(chunk)
            data = [
                {
                    "doc_id": hashed_doc_id, 
                    "passage_id": passage_id, 
                    "domain": domain, 
                    "title": title, 
                    "text": chunk, 
                    "text_emb": chunk_emb, 
                    "info": info, 
                    "tags": tags
                }
            ]        
            self.vectordb.insert_data(data, collection_name=domain)
    
    def retrieve_data(self, query, top_k, domain, output_fields='text'):
        cleansed_text = self.data_p.cleanse_text(query)
        query_emb = self.emb_model.bge_embed_data(cleansed_text)
        collection = Collection(domain)
        self.vectordb.set_search_params(query_emb, limit=top_k, output_fields=output_fields)
        search_result = self.vectordb.search_data(collection, self.vectordb.search_params)
        text = self.vectordb.decode_search_result(search_result)
        print(text)
        return text