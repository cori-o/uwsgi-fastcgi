# uwsgi-fastcgi-milvus-api
A FastCGI-based setup for running Flask with uWSGI and Nginx using Docker Compose

### Docker compose
#### 1. Customize docker-compose.yml if needed 
- Ensure that the Flask and Nginx services share the volume (shared_tmp) and network (flask_network)
- Modify docker-compose.yml as necessary


#### 2. Start Containers 
```bash
$ docker-compose up -d --build
```
- The -d flag runs the containers in the background
- The --build flag rebuilds the containers if changes are made

#### 3. edit ./rag/.env 
We need to set the ip_addr value in the ./rag/.env file according to our development environment in order to connect to the Milvus container

### RAG Milvus Setting
#### 1. Set collection
Enter the milvus-manager container and modify set_collection.py if needed:
```
$ docker exec -it milvus-manager /bin/bash
$ vi set_collection.py
```

You can create a new collection by running:
```
$ python set_collection.py --collection_name [collection_name]
```

#### 2. check Milvus
Use check_milvus.py to view collection/partition information, or to create/delete partitions.
```
$ python check_milvus.py --collection_name [collection_name]   # get collection, partition info
$ python check_milvus.py --task_name create --collection_name [collection_name] --partition_name [partition_name]   # create partition
$ python check_milvus.py --task_name delete --collection_name [collection_name] --partition_name [partition_name]   # delete partition
$ python check_milvus.py --task_name drop --collection_name [collection_name]   # drop collection 
```


### Verify the Setup
```
$ curl http://localhost
```
Expected output: Hello, FastCGI is working!

! nginx and uwsgi must share a volume to connect via a Unix socket
Flask is executed through uwsgi, so ensure that uwsgi.ini or the uwsgi execution command is correctly configured
If Nginx returns a 502 Bad Gateway error, verify that uwsgi.sock is properly shared
