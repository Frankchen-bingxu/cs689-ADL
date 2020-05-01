# CS636 Deep Learning
NAME:CHEN BINGXU UIN:329006933.  
Classify videos using  C3D and Dense ResNet

## topic
detect crawling(baby) , I train the model on two dataset to test the accuracy and the process and how to test the video will show in the following.
## Requirements
tensorflow
opencv
keras

## The fold of the project
I have train the model first, you don't need to train model anymore.In this project, open fold c3d, you can see three fold c3d,res3d. And the test videos are in video_test fold,the json files(contain time/label) are shown in the fold.  
## How to run
use command to run
### 1
cd c3d/c3d.  
python video_demo.py
### 2   
run video_demp.py to test the video in the video_test fold(maybe you need to change the path if you want to change the video to test or test the video you provid). Also you can generate your own json file.

## The video of how to run
https://youtu.be/ayIpQWEYsms
