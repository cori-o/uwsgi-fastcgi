# uwsgi-fastcgi-rag
A FastCGI-based setup for running Flask with uWSGI and Nginx using Docker Compose, enhanced with Milvus and RAG

### Docker compose
#### 1. Customize docker-compose.yml if needed 
- Ensure that the Flask and Nginx services share the volume (shared_tmp) and network (flask_network).
- Modify docker-compose.yml as necessary.


#### 2. Start Containers 
```bash
$ docker-compose up -d --build
```
- The -d flag runs the containers in the background.
- The --build flag rebuilds the containers if changes are made


### Check Docker status 
```bash
$ docker ps -a
```
After running docker-compose up, both the nginx-container and flask-container should be up and running\
e.g) Up 10 seconds

#### !!(optional) Kill all container 
Once you have finished using them, you can stop the running containers with the following command
```
docker-compose down -v
```


### Verify the Setup 
```
$ curl http://localhost
```
Expected output: Hello, FastCGI is working!

! nginx and uwsgi must share a volume to connect via a Unix socket.
Flask is executed through uwsgi, so ensure that uwsgi.ini or the uwsgi execution command is correctly configured.
If Nginx returns a 502 Bad Gateway error, verify that uwsgi.sock is properly shared.


### Memory Usage Check 
We can check the memory usage of each Docker container by running the following command:
```
docker stats --no-stream
docker stats --no-stream flask-container
```
If you notice a container consuming a large amount of memory, you can check the processes using the most memory inside the container by running the following command:
```
docker exec -it [container-id] /bin/bash
ps -aux --sort=-%mem | head -20
```
