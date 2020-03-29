#!/bin/sh
python train.py \
  --train_set_path=dataset/iris_training.csv \
  --test_set_path dataset/iris_test.csv \
  --model_dir=models
