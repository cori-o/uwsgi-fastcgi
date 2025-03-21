from flask import Flask, send_file, request, jsonify, Response
from pymilvus import Collection
from dotenv import load_dotenv
from src import EnvManager, DBManager
import logging
import json 
import os 

logger = logging.getLogger('api_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('api-result.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

load_dotenv()
args = dict()
args['config_path'] = "./config"
args['llm_config'] = "llm_config.json"
args['db_config'] = "db_config.json"
args['collection_name'] = "news"
args['ip_addr'] = os.getenv('ip_addr')   

env_manager = EnvManager(args)
env_manager.set_processors()
emb_model = env_manager.set_emb_model()
milvus_data, milvus_meta = env_manager.set_vectordb()
milvus_db = env_manager.milvus_db
interact_manager = DBManager(data_p=env_manager.data_p, vectorenv=milvus_db, vectordb=milvus_data, emb_model=emb_model)


@app.route('/data/show', methods=['GET'])
def show_data():
    '''
    "collection_name": "news"   # news, description, etc ... 
    '''
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({"error": "collection_name is required"}), 400
    
    try:
        assert Collection(collection_name)
    except: 
        response_data = {"error": "유효한 Collection 이름을 입력해야 합니다.", "collection list": milvus_db.get_list_collection()}
        return Response(json.dumps(response_data, ensure_ascii=False), content_type="application/json; charset=utf-8")
    
    milvus_db.get_collection_info(collection_name)
    milvus_db.get_partition_info(collection_name)               
    return jsonify({
        "schema": milvus_db.collection_schema.to_dict(),
        "partition_names": milvus_db.partition_names,
        "partition_nums": milvus_db.partition_entities_num,
    }), 200

@app.route('/data/insert', methods=['POST'])
def insert_data():
    '''
    data: {
        "jobID": 
    
    }
    '''
    data = request.json
    return jsonify({"status": "received"}), 200


@app.route('/data/delete', methods=['POST'])
def delete_data():
    '''
    data: {
        "jobID":
    }
    '''
    data = request.json
    return jsonify({"status": "receviced"}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello, FastCGI is working!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
