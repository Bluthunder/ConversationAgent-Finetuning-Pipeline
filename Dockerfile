FROM lightningai/pytorch:2.1-cuda11.8-cudnn8-runtime

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
