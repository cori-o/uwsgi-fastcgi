# uwsgi-fastcgi
A FastCGI-based setup for running Flask with uWSGI and Nginx using Docker Compose.

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


#### 3. Editing Nginx Configuration
we need to edit default.conf file of nginx in container 
```bash
$ docker exec -it nginx-container /bin/bash 
```
Edit default.conf
```bash
$ apt-get update -y
$ apt install vim -y
$ cd /etc/nginx/conf.d
$ vi default.conf
```
Replace the content of default.conf with the configuration from this repository.
Then restart nginx 
```bash
$ docker-compose restart nginx
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
