from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from dotenv import load_dotenv
from src import milvus 
import os 
import argparse
import json

def main(args):
    load_dotenv()
    ip_addr = os.getenv('ip_addr')

    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)
    db_args['ip_addr'] = ip_addr

    milvus_db = milvus.MilvusEnvManager(db_args)
    print(milvus_db.get_list_collection())

    if args.task_name == None: 
        ''' Collection, Partition info 조회 ''' 
        print(f'[Info] collection {args.collection_name}')
        milvus_db.get_collection_info(args.collection_name)
        print(f'[Info] partition')
        milvus_db.get_partition_info(args.collection_name)
        print(dict(zip(milvus_db.partition_names, milvus_db.partition_entities_num))) 
    elif args.task_name == 'create':   # create partition 
        try:
            assert args.partition_name != None, "생성하고자하는 partition 이름을 지정해주세요."
            collection = Collection(args.collection_name)
            collection.load()
            milvus_db.create_partition(collection, args.partition_name)
        except:
            pass 
    elif args.task_name == 'delete':
        try: 
            assert args.partition_name != None, "삭제하고자하는 partition 이름을 지정해주세요."
            milvus_db.delete_partition(args.collection_name, args.partition_name)
        except:
            pass
    elif args.task_name == 'drop':
        try:
            assert args.collection_name != None, "삭제하고자하는 collection 이름을 지정해주세요."
            milvus_db.delete_collection(args.collection_name)
        except:
            pass
    else:
        print(f'create, delete 작업이 가능합니다.')

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--collection_name', type=str, required=True)
    cli_parser.add_argument('--partition_name', type=str, default=None)
    cli_parser.add_argument('--task_name', type=str, default=None)
    cli_args = cli_parser.parse_args()
    main(cli_args)