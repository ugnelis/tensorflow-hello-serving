FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get update \
    && pip install --upgrade pip

WORKDIR /hello

COPY / .

RUN pip install -r requirements.txt
