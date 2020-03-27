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
$ docker image build -t hello-serving:1.0 .
```

### Run Docker Container

Run a Docker container together with NVidia GPUs:
```bash
$ docker run -it --gpus all --rm hello-serving:1.0 bash
```

Read more about it in [Docker Hub](https://hub.docker.com/r/tensorflow/tensorflow/).
