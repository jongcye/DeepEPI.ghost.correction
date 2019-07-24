# EPI_ghost_learning

This code is a EPI ghost correction code using deep learning. 

## Paper
Juyoung Lee, Yoseob Han, Jae-Kyun Ryu, Jang-Yeon Park and Jong Chul Ye, "k-Space Deep Learning for Reference-free EPI Ghost Correction", Magnetic Resonance in Medicine (in press), 2019


## Requirements
* The codebase is implemented in Matlab.
* MatConvNet (matconvnet-1.0-beta24) 
  * Download the 'function' folder, and move the files on MatConvNet folder.
  
## Dataset
* The whole data used in the paper are private data, so only some sample data are uploaded here.
* The data are 3T MR EPI brain image. Input data is a ghost image, and the label data is a free-ghost image. For the label data, ghost is removed by using ALOHA.

## Training
* Main file to train is 'main_ghost_learning.m'.
  * Various learning parameter(e.g. learning rate, # of epochs) can changed on this main file.
* There are some sample data in 'db' folder for training. The number of channel of input data is 2*coil. 
* You can change the filter size, network depth in 'cnn_ghost_init.m'

## Inference
* To inference with trained model, run display_cnn_ghost.m
* There are some sample data in 'db' folder for inference.
