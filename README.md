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

#### 3. edit ./rag/.env 
You need to set the ip_addr value in the .env file according to your development environment in order to connect to the Milvus container.

### Verify the Setup 
```
$ curl http://localhost
```
Expected output: Hello, FastCGI is working!

! nginx and uwsgi must share a volume to connect via a Unix socket.
Flask is executed through uwsgi, so ensure that uwsgi.ini or the uwsgi execution command is correctly configured.
If Nginx returns a 502 Bad Gateway error, verify that uwsgi.sock is properly shared.
