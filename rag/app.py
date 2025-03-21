from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from flask import Flask, send_file, request, jsonify, Response
from src import MilVus, DataMilVus, MilvusMeta
from src import OpenAIQT, RulebookQR
from src import EmbModel, LLMOpenAI
from src import EnvManager, ChatUser
import tempfile
import logging
import json
import os

logger = logging.getLogger('stt_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('stt-result.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)
args = dict()
args['config_path'] = "./config"
args['llm_config'] = "llm_config.json"
args['db_config'] = "db_config.json"
args['collection_name'] = "rule_book"
env_manager = EnvManager(args)
env_manager.set_config()
print(env_manager)

data_milvus = env_manager.set_vectordb()
milvus_db = env_manager.milvus_db
emb_model, response_model = env_manager.set_llm()
print(emb_model, response_model)


@app.route('/data/show', methods=['GET'])
def show_data():
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({"error": "collection_name is required"}), 400
    
    collection = Collection(collection_name)
    milvus_db.get_partition_info(collection)
    milvus_db.get_collection_info(collection_name)
    # 조회 결과를 JSON 형태로 반환
    return jsonify({
        "collection_name": collection_name,
        "partition_info": milvus_db.partition_names,
        "collection_info": milvus_db.partition_entities_num
    }), 200

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello, FastCGI is working!"})

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
