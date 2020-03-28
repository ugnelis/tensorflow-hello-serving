# TensorFlow Hello Serving

Simple example of TensorFlow 1.x serving.

## Training

Model training:
 
```bash
$ ./train.sh
```

## Prediction

Prediction using the trained model:
 
```bash
$ ./predict.py
```

## Development (with Docker)

### Build Docker Image

In order to run this project as a Docker container, an Docker image has to be built:
```bash
$ docker build --file Dockerfile.development -t hello-development:1.0 .
```

### Run Docker Container

Run a Docker container together with NVidia GPUs:
```bash 
$ docker run -it \
    --gpus all \
    --name hello-development \
    hello-development:1.0 bash
```

In case you need to take your trained models out of the container:
```bash 
$ docker cp docker-development:/workspace/saved_model/ received/
```

Read more about it in [Docker Hub](https://hub.docker.com/r/tensorflow/tensorflow/).

## Production (with Docker)

### Build Docker Image

Docker Image build which contains of TensorFlow serving:
```bash
$ docker build --file Dockerfile.production -t hello-production:1.0 .
```

### Run Docker Container

Run a Docker container:
```bash
$ docker run -it --rm \
    -p 8501:8501 \
    --name hello-production \
    hello-production:1.0
```
