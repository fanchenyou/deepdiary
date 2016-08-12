#!/bin/bash

source global.sh

TEST_DATA_DIR=$EXP_ROOT/test_data
TEST_IMAGE_DIR=$TEST_DATA_DIR/imgs


# Model file

coco_raw=$TRAINED_MODEL_ROOT/caffe_lstm_COCO.p
caffe_finetune=$TRAINED_MODEL_ROOT/caffe_lstm_finetune_512_2_layer.p


#MODELS=(caffe_finetune_512_2)
MODELS=(coco_raw caffe_finetune)

cd $PRED_ROOT

# test Diverse M-Best Solutions 


for model in ${MODELS[@]}
do
  python predict_on_images_html.py ${!model} -k $TEST_DATA_DIR/result_${model}.html  -r $TEST_DATA_DIR --fix_diverse 1 --beam_size -2 
done
