#!/bin/sh
# From the test set: 6.4,2.9,4.3,1.3,1
python predict.py \
  --model_dir models \
  --inputs "6.4,2.9,4.3,1.3"
