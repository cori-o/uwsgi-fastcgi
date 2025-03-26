from flask import Flask, send_file, request, jsonify, Response
from pymilvus import Collection
from dotenv import load_dotenv
from src import EnvManager, InteractManager
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
args['collection_name'] = "congress"
args['ip_addr'] = os.getenv('ip_addr')   

env_manager = EnvManager(args)
env_manager.set_processors()
emb_model = env_manager.set_emb_model()
milvus_data, milvus_meta = env_manager.set_vectordb()
milvus_db = env_manager.milvus_db
interact_manager = InteractManager(data_p=env_manager.data_p, vectorenv=milvus_db, vectordb=milvus_data, emb_model=emb_model)

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

@app.route('/search', methods=['GET'])
def search_data():
    query_text = request.args.get('query_text')
    top_k = request.args.get('top_k', 5)
    domain = request.args.get('domain')

    if not domain:
        return jsonify({"error": "유효한 도메인 이름을 입력해주세요"}), 400  # 400 Bad Request
    try:
        top_k = int(top_k)
    except ValueError:
        return jsonify({"error": "top_k 값은 숫자여야 합니다."}), 400
    
    text = interact_manager.retrieve_data(query_text, top_k, domain)
    print(f"Search results: {text}")
    response_data = {
        "message": "data search complete !",
        "query_text": query_text, 
        "top_k": top_k, 
        "domain": domain, 
        "results": text
    }
    return Response(json.dumps(response_data, ensure_ascii=False), content_type="application/json; charset=utf-8")
    
@app.route('/insert', methods=['POST'])
def insert_data():
    '''
    doc_id: yyyymmdd-title   e.g) 20240301-메타버스 뉴스
    data: {
        "domain": "news"   - collection_name 
        "title": "메타버스 뉴스"
        "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다 ... "
        "info": {
            "press_num": "비즈니스 워치"
            "url": "http://~"
        }
        "tags": {
            "date": "20220804"
            "user": "user01"
        }
    }
    '''
    data = request.json
    doc_id = data['tags']['date'].replace('-','') + '-' + data['title']
    if data['domain'] not in milvus_db.get_list_collection():
        interact_manager.create_domain(data['domain'])
    interact_manager.insert_data(data['domain'], doc_id, data['title'], data['text'], data['info'], data['tags'])
    return jsonify({"status": "received"}), 200

@app.route('/delete', methods=['DELETE'])
def delete_data():
    '''
    data: {
        "date": "20220804"
        "title": "메타버스%20뉴스"
        "domain": "news"
    }
    '''
    doc_date = request.args.get('date')
    doc_title = request.args.get('title')
    doc_domain = request.args.get('domain')
    doc_id = doc_date.replace('-','') + '-' + doc_title
    interact_manager.delete_data(doc_domain, doc_id)
    return jsonify({"status": "receviced"}), 200

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello, FastCGI is working!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
