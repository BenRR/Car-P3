**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: samples/nvidia-model.png "NVidia Model"
[image2]: samples/center.png "center"
[image3]: samples/edge1.png "Recovery Image 1"
[image4]: samples/edge2.png "Recovery Image 2"
[image5]: samples/edge3.png "Recovery Image 3"
[image6]: samples/flipped1.png "Normal Image"
[image7]: samples/flipped2.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* data_augment.py containing the script to generate training data set from the raw driving log and images.
* train.py containing the script to create and train the model
* track1-best.h5 containing a trained convolution neural network for track 1
* track1-finish.mp4 track 1 model driving through video
* track2-best.h5 containing a trained convolution neural network for track 2
* track2-finish.mp4 track 1 model driving through video
* writeup.md summarizing the results
* drive.py for driving the car in autonomous mode
* video.py fro creating the submission video

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

For track 1
```sh
python drive.py track1-best.h5
```

For track 2
```sh
python drive.py track2-best.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I choose the nVidia model as base architecture.(train.py lines 112-161)
On top of the model I added image cropping (train.py line 104), image normalization (train.py line 107) and many other normalization and dropout layers in between convolution and FC layers.

#### 2. Attempts to reduce overfitting in the model

The model contains 4 dropout layers in order to reduce overfitting (train.py lines 117, 125, 133, 142).

I have created generator for batch the training (train.py line 59)

The model was trained and validated on different data sets to ensure that the model was not overfitting (train.py code line 88). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 174).
Several hyper-parameters defined between train.py line 33 and 37
Also I have tuned the drop out rate train.py line 94

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And in data_augment.py I use multi-cam images also flipped the images to generate more training data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the nVidia self-driving car model I thought this model might be appropriate because it is powerful enough to solve this problem.

In order to gauge how well the model was working,
* I generated more data by using multi-cam and flipped images
* I implemented the nVidia model but only added one dropout layer before FC layers
* I also added checkpoints for every improvement of validating loss (train.py line 170)
* The best model I got from the first attempt with both low training and validation loss however it performs poorly in track1 driving
* After reviewing several early checkpoints I found some early epochs result worked much better for track1.
* I suspected there could be a overfitting so I added several more dropout layers
* After that I had a much better model which could drive through most part of track1 but failed on the dirt road turn after the bridge
* Then I used the emulator to record more frames around the problematic area
* Then I trained the best previous model against these new frames for a couple of epochs
* Finally my model can successfully finish track 1
* To make it work for track 2, I had to record more images on track 2 and keep training the model for around another 10 epochs
* Then I got another model that worked for track 2

#### 2. Final Model Architecture

The final model architecture (train.py lines 101-161) consisted of a convolution neural network with the following layers
* conv layer (24, 5, 5, same, elu, maxpool(5,5,same), normalization)
* dropout
* conv layer (36, 5, 5, same, elu, maxpool(5,5,same), normalization)
* dropout
* conv layer (48, 3, 3, same, elu, maxpool(3,3,same), normalization)
* dropout
* conv layer (64, 3, 3, same, elu, maxpool(3,3,same), normalization)
* dropout
* fully connected (1164) elu
* fully connected (100) elu
* fully connected (50) elu
* fully connected (10) elu
* output

![nvidia-model][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][image2]

I then recorded two laps of the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from side of the road back to center. These images show what a recovery looks like starting from ... :

![recover1][image3]

![recover2][image4]

![recover3][image5]

I then did the above again anti-clockwise on track one.

Then I repeated whole process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![flipped-1][image6]

![flipped-2][image7]

After the collection process, I had 100,000 number of data points.
In my code there are more processes like shuffling, cropping and normalization (train.py line 62, 79, 104, 107)

I finally randomly shuffled the data set and put 40% (train.py line 35) of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by validation loss not improving anymore.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
