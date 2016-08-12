#!/bin/bash

source ~/.profile
source ~/global.sh

#CAFFE_ROOT=/home/fan6/lstm/caffe-caption

#final model name
HIDDEN_SIZE=512
LAYER=2


PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH

FINAL_MODEL=models/caffe_lstm_finetune_"$HIDDEN_SIZE"_"$LAYER"_layer.p


python finetune.py \
-d1 'coco' -d2 'lifelog' \
--deviceId '2' \
--hidden_size $HIDDEN_SIZE \
--num_layer 2  \
--max_len 30 \
--word_count_threshold 20 \
--batch_size 32 \
--grad_clip 5 \
--solver 'rmsprop' \
--save_model_as $FINAL_MODEL \
--snapshot_dir 'snapshot' \
--lifelog_only 0
