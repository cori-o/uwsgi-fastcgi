from src import milvus, MilvusEnvManager
from dotenv import load_dotenv
import argparse
import json
import os 


def main(args):
    load_dotenv()
    ip_addr = os.getenv('ip_addr')

    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)
    db_args['ip_addr'] = ip_addr

    milvus_db = MilvusEnvManager(db_args)
    print(f'ip: {milvus_db.ip_addr}')

    milvus_db.set_env()
    print(f'client: {milvus_db.client}')
    
    '''
    customize your field schema
    '''
    data_doc_id = milvus_db.create_field_schema('doc_id', dtype='VARCHAR', is_primary=True, max_length=1024)
    data_passage_id = milvus_db.create_field_schema('passage_id', dtype='INT64')
    data_domain = milvus_db.create_field_schema('domain', dtype='VARCHAR', max_length=32)
    data_title = milvus_db.create_field_schema('title', dtype='VARCHAR', max_length=128)
    data_text = milvus_db.create_field_schema('text', dtype='VARCHAR', max_length=512)   # 500B (500글자 단위로 문서 분할)
    data_text_emb = milvus_db.create_field_schema('text_emb', dtype='FLOAT_VECTOR', dim=1024)
    data_info = milvus_db.create_field_schema('info', dtype='JSON')
    data_tags = milvus_db.create_field_schema('tags', dtype='JSON')
    schema_field_list = [data_doc_id, data_passage_id, data_domain, data_title, data_text, data_text_emb, data_info, data_tags]

    schema = milvus_db.create_schema(schema_field_list, 'schema for rag, using fastcgi')
    collection = milvus_db.create_collection(args.collection_name, schema, shards_num=2)
    milvus_db.get_collection_info(args.collection_name)
    milvus_db.create_index(collection, field_name='text_emb')   # doc_id 필드에 index 생성 

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--collection_name', type=str, default=None)
    cli_argse = cli_parser.parse_args()
    main(cli_argse)