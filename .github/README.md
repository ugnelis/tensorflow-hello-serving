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
$ ./predict.sh
```

## Development (with Docker)

### Build Docker Image

In order to run this project as a Docker container, a Docker image has to be built:
```bash
$ docker build --file Dockerfile.development -t hello-development:1.0 .
```

### Run Docker Container

Run a Docker container together with NVidia GPUs:
```bash 
$ docker run -it \
    -v ${PWD}/exported_models:/workspace/exported_models \
    --gpus all \
    --name hello-development \
    hello-development:1.0 bash
```

In case you need to take your trained models out of the container:
```bash 
$ docker cp docker-development:/workspace/models/ models/
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
    -v ${PWD}/exported_models:/models \
    -p 8501:8501 \
    -e MODEL_NAME=hello \
    -e MODEL_PATH=/models/hello \
    --name hello-production \
    hello-production:1.0
```

### Send Request

When production Docker container is running, the prediction requests can be done:

[**POST**] "http://localhost:8501/v1/models/hello:predict"

JSON body example:
```json
{
  "signature_name": "predict",
  "instances": [
    {
      "f1": 6.4,
      "f2": 2.9,
      "f3": 4.3,
      "f4": 1.3
    }
  ]
}
```
