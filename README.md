# Lyft Challenge
## Image Segmentation using VGG net in Keras.

Inspired from https://github.com/divamgupta/image-segmentation-keras

Implementation of Deep Image Segmentation model for Lyft challenge in keras. 

<p align="center">
  <img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png" width="50%" >
</p>


## Models 
* VGG Segnet 

## Getting Started

### Prerequisites

* Keras 2.0
* opencv for python

```shell
sudo apt-get install python-opencv
sudo pip install --upgrade keras
sudo pip install pydot
sudo pip install graphviz
sudo apt install graphviz
```

### Preparing the data for training

mkdir data
cd data
wet https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz -O dataset.tar.gz
tar -xvzf dataset.tar.gz

## Visualizing the prepared data

You can also visualize your prepared annotations for verification of the prepared data.

```shell
./run.sh visualize
```



## Downloading the Pretrained VGG Weights

You need to download the pretrained VGG-16 weights trained on imagenet if you want to use VGG based models

```shell
mkdir data
cd data
wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5"
wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
```

## Training the Model

To train the model run the following command:

```shell
./run.sh train
```

## Getting the predictions

To get the predictions of a trained model

```shell
./run.sh predict <model_id>
```
