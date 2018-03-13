
# **Traffic Sign Recognition** 

## Nick Xydes Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/barchart.png "Distribution of classes in dataset"
[image2]: ./examples/example1.png "Example PreProcessing 1"
[image3]: ./examples/example2.png "Example PreProcessing 2"
[image4]: ./examples/GermanSigns.png "German Signs"
[image5]: ./examples/GermanSignsProcessed.png "German Signs Pre-Processed"
[image6]: ./examples/Softmax.png "Softmax Probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Submission Files

#### 1. Ipython notebook with code

Here is a link to my [project code](https://github.com/nxydes/P2-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)
#### 2. Writeup report
You're reading it!

### Data Set Exploration

#### 1. The submission includes a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (32 x 32 pixels with 3 color channels)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The following bar chart shows how much of each type of image is in the training data set. You can see the distribution of types is not even with some classes of images having over 10x as many as other classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1.  Describes the preprocessing techniques used and why these techniques were chosen.

Before I do anything else I actually shuffle the input, this way the data is not in order of image class. This will prevent the data from overfitting a certain class during the training.

First, I chose to convert the image to grayscale. From what I've seen in the lectures and in reading about photo classification online this step seems to improve the results. It also helps to cut down on the memory needed as it immediately cuts the size of the first filter down by 3.

Second, I noticed many of the images were very very dark or sometimes overly light. So I apply a histogram equalization algorithm from the CV2 library. There were two possibilities for this so I tried them both, one adjusts the contrast globally in the image, the other uses a filter of 4x4 or 8x8 pixels to do it locally in the image. The smaller filter should result in a better result but I found the image ended up too pixelated and while it resulted in better validation accuracy it resulted in worse accuracy on the internet images, so I stuck with the global contrast adjustment.

I noticed a lot of examples of people applying random rotations and skews onto their data set. I experimented with this but believe I probably needed to use this by adding more data to the dataset instead of rotating the existing data. I disabled it for the final run.

As a last step I normalized the image data to provide a better input for the training algorithm. This is described in the lectures as helping with the training speed and performance.

![alt text][image2]

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image  					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6  			     	|
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120        							|
| RELU					|         										|
| Dropout				| Drop 50% during training						|
| Fully connected		| Outputs 84									|
| RELU					| 												|
| Dropout				| Drop 50% during training						|
| Fully connected		| Outputs 43									|
 
This is very similar to the example LeNet architecture. I was able to improve the performance of the LeNet example by better pre-processing the images and by adding the Dropout layers. I added the dropout layers based on the earlier discussion in the lecture series that said if your architecture is over fitting the data then the dropout layers can help. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used the same approach as in the LeNet example. However, I used a very small batch size of 70 and stopped training at 25 epochs. I found a small training rate of 0.0009 to be the best and maintained a sigma of 0.1 and mu of 0. I also had a keep percentage for the dropout layers of 50% during training, but during evaluation this was obviously put back at 100%. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 97.3%
* test set accuracy of 95.6%

I chose an iterative approach to solve this problem as I have very little experience with deep learning and wanted to go through an iterative process to teach myself about the different controls and tuning parameters. I started with the lenet lab architecture as it was what was taught to us in the lecture series. It seemed like a perfect starting point. 

Initially the architecture seemed to be overfitting as I had a gap of over 10% between the training accuracy and the validation accuracy. I started playing with parameters and uses some online resources to get tips on which parameters to tune. Eventually, the best I could do without dropout was a validation accuracy of 0.91 with a training accuracy of 0.995. While this was ok, I wanted to do better. I added dropout and kept iterating. I tried large numbers of epochs but found it didn't improve much after 25. I played with the batch size making it as large as 500 or as small as 30 but eventually settles on 70. The rate I didn't play with too much, aside from lowering it a tad based on advice from some articles I came across related to image classification.

The end results were satisfactory for my first attempt at this problem with over 97% validation accuracy. Adding the dropout layers is what took my model from below passing to above satisfactory. This was clearly super important in making sure I wasn't over fitting the dataset. This is actually the point of dropout layers, to make sure that the model can't rely on any one connection in the neural network to be there, it has to make multiple connections for each feature to ensure it exists after the dropout layer which results in a more robust total system.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found 7 german signs on the web that I decided to use. I chose these in a few ways because I wanted to see how my network would perform against what I perceived to be hard and easy challenges. 
![alt text][image4]
![alt text][image5] 

I thought to myself the speed limit signs would be difficult to classify due to their similarity to themselves and other signs, the only difference being the numbers themselves. I chose a 50, 20 and 30 speed limit sign for my test set. The 50 kph sign might be difficult because it goes all the way up to the edge of the image. The 20 sign may be difficult because I realized after I downloaded it that the sign itself is missing the outer circle of white, so its not actually the same as the others. The 30 kph sign should be the easiest sign to get.

I then chose 3 arrow signs. I got a keep right sign, turn right ahead sign and roundabout sign. These three I anticipated would be easy to clasify as they each are well defined, except there is a smudge over the roundabout sign which may cause problems?

Finally, I chose a stop sign. I am not sure how this one would work, I figure it should be easy due to the fact its unlike most other signs in the classified set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| Correct			|
|:---------------------:|:---------------------------------------------:| :----------------:|
| Roundabout mandatory 	| Priority Road									| 0					|
| Speed Limit 50km/h	| Speed Limit 30km/h							| 0 				|
| Keep Right			| Keep Right									| 1                 | 
| Stop		      		| Stop							 				| 1					|
| Speed Limit 20km/h	| Keep Left		    							| 0					|
| Turn right ahead		| Turn right ahead								| 1					|
| Speed Limit 30km/h	| Speed Limit 30km/h							| 1					|

Overall my model guessed 4 out of 7 correctly for a total accuracy of 57.1%. I believe the discrepency with the test and validation accuracy is due to the difference in the images I pulled from the web. One of the images neglected to have the white border around the speed limit sign and it incorrectly guessed that image class. The roundabout sign also had occlusion on the sign itself. Finally, the 50km/h sign was correctly guessed to be a speed limit sign but got 3 instead of 5, which are two numbers that are close together in shape. I can see why the mistakes are made which should help me to improve on the performance should I be inclined and had enough time.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The best way to visualize this is to show it graphically. I plotted each input image on the right with the top 5 softmax probabilities to the right of it with exemplary images to compare, with percentages in the labels.

![alt text][image6]

The model is, to put it bluntly, hilariously confident in its decisions regardless of how good of a decision it makes. For each image it guessed correctly, it has a softmax probability of 100% for that prediction, but for the 3 it guessed incorrectly, it had 100%, 98% and 93% confidence in its incorrect guess. 


