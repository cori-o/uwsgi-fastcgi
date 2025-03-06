# 기본 Docker Image 설정   
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel    
LABEL description='Docker image created for flask'

# Docker 내 작업 경로 설정 
WORKDIR /app 
COPY . .

# 패키지 설치 (한글 입력을 위한 locale)
RUN apt-get update && apt-get install -y locales && apt-get install -y build-essential
RUN apt-get install curl -y
RUN apt-get install python3-pip -y
RUN pip install --no-cache-dir uwsgi
RUN pip install -r requirements.txt
CMD ["uwsgi", "--socket", "0.0.0.0:9010", "--protocol=fastcgi", "--wsgi-file", "app.py", "--callable", "app"]