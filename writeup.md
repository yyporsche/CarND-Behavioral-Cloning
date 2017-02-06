#**CarND-Behavioral-Cloning** 
---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train_reguression.py the main script to train
* model.py containing the script that defines different models
* process.py implements several image processing methods for different models
* data.py implements a DataSet class that provides a customized batch generator to model
* config.py includes preprocessing, SDC model and training process
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submssion includes functional code
drive.py is modified to load the customized SDC model, the car can be driven autonomously around the track by executing 
```sh
python -m sdc.drive
```

####3. Submssion code is usable and readable

The train_reguression.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I used pre-trained VGG16 model and fine tuned the last two layers with the training data from simulator.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data released from Udacity.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started the project from this [github repo](https://github.com/dolaameng/Udacity-SDC_Behavior-Cloning)

I originally tried the VGG16 pretrained approach and run the training with 20 epoches only using the center image, this model gave a pretty good starting point, but the turning steer seems to be not sharp enough and the car will drive out of the road after the bridge in the first track. I looked at the plot between the real steer angle and predict steer angle:
![VGG16 with 20 epochs using only center images](https://github.com/yyporsche/CarND-Behavioral-Cloning/blob/master/pics/inspection_vgg_ori_epoch_20.png)

Then I added the left and right camera data, add 0.25 steer angle to left and minus 0.25 steer angle to right, in this case I will have more corner data. This time I run the training with 10 epoches first and the prediction is:
![VGG16 with 10 epochs using center/left/right images](https://github.com/yyporsche/CarND-Behavioral-Cloning/blob/master/pics/inspection_vgg_addsideimage_epoch_10.png)

At last I trained the model with 10 more epochs:
![VGG16 with 20 epochs using center/left/right images](https://github.com/yyporsche/CarND-Behavioral-Cloning/blob/master/pics/inspection_vgg_addsideimage_epoch_20.png)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 77-103) consisted of a pretrained VGG16 from block5_conv3 followed with three fully connected layers with size of 4096/2048/1024 and finally converge to the prediction steer angle.

####3. Creation of the Training Set & Training Process

I used the data from Udacity. I filter out the data points which speed is less than 20. 

After the collection process, I had X number of data points. I then preprocessed this data by ...

I added the left and right camera data into the training with a 0.25 steer degree offset. Then I mirrored all the data to get another full data set. I finally randomly shuffled the data set and put 10% of the data into a validation set and 10% of the data into a test set.
