# 기본 Docker Image 설정   
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
LABEL description='Docker image created for milvus-api architecture'

# Docker 내 작업 경로 설정 
WORKDIR /milvus-api 
COPY . .

# 한글 입력을 위한 환경 변수 설정 
ENV LC_ALL ko_KR.UTF-8 

# 패키지 설치 (한글 입력을 위한 locale)
RUN apt-get update && apt-get install -y locales && apt-get install -y build-essential
RUN apt-get install curl -y
RUN locale-gen ko_KR.UTF-8   
RUN apt-get install python3-pip -y
RUN pip install -r requirements.txt
RUN pip install -U FlagEmbedding
CMD ["uwsgi", "--socket", "0.0.0.0:9010", "--protocol=fastcgi", "--wsgi-file", "app.py", "--callable", "app"]