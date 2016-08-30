# Deepdiary
This repository is [caffe](http://caffe.berkeleyvision.org/) implementation of image captioning on lifelogging data.
Please see our paper([1](https://arxiv.org/abs/1608.03819)) for information of how to use this package on your own dataset.

The core of this code is 
- src/caffe/layers/caption_input_layer.cu
- src/caffe/layers/lstm_layer.cu
- src/caffe/layers/lstm_loss_layer.cu

## Lifelogging dataset
We publish our lifelogging dataset by releasing image features along with human labelings. You can download them [here](http://vision.soic.indiana.edu/deepdiary_files/data.zip) (image features and sentences).

We tend to not publish the whole dataset with real photos for the reason to protect privacy.
However, we do publish a subset of the dataset with photos which we published on Amazon Mechanical Turk for public labeling. You can download them [here](http://vision.soic.indiana.edu/deepdiary_files/amt_data.zip). 
For more details of how the dataset is collected, please refer to our [paper](https://arxiv.org/abs/1608.03819).

## Training
To run training demo, 
- clone this repository to your directory, say $CAFFE_ROOT
- download our packed lifelogging data and extract them as examples/myexp/data
- download VGG_ILSVRC_16_layers.caffemodel from https://github.com/BVLC/caffe/wiki/Model-Zoo and place it into $CAFFE_ROOT/models/VGG_ILSVRC_16_layers
- cd examples/myexp
- change setting.py home_dir, global.sh CAFFE_ROOT to same folder that you clone this repo.
- ./run_finetune.sh

You can also skip training stage, and use our trained models to do prediction
```
cd examples/myexp/models
unzip models
```

## prediction
To run testing demo,
```
cd examples/myexp/test_data, check imgs folder, tasks.txt, and vgg_feats.mat
cd ..
./predict.sh
```

To predict on you own images, 
- replace imgs folder with your images, and output names in tasks.txt, one name per line
- extract VGG features for each image in order as image names in tasks.txt, name as vgg_feats.mat file, for example, there are 7 images in current imgs folder, so vgg_feats.mat contains an 'feat' entry which is a matrix of size 4096x7
- choose your model file in predict.sh. This demo provides two pretrained models by authors. `coco_raw` is a model trained with COCO data, `caffe_finetune` is finetuned by lifelogging data. You will see that `caffe_finetune` produces much better predictions with diverse structures
- explore examples/myexp/prediction folder, the M-Best diverse technique is implemented in lstm_generator.py

# LSTM formulation
This implementation strictly follows "Show and Tell: A Neural Image Caption Generator" Eq(4~9)
https://arxiv.org/abs/1411.4555


# Going deep
Input takes, 
- json file which contains imageId and sentences as ground truth, which is congruent to Karpathy's neuraltalk ([repo] https://github.com/karpathy/neuraltalk)
- image features extracted from a CNN (typically bvlc_reference_net or vgg_net)



# License and Citation

Please cite the following paper in your publications if you use our dataset or code in your research:

    @inproceedings{deepdiary2016eccvw,
      title = {DeepDiary: Automatically Captioning Lifelogging Image Streams},
      author = {David Crandall and Chenyou Fan},
      booktitle = {European Conference on Computer Vision International Workshop on Egocentric Perception, Interaction, and Computing}
    }
    
    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

