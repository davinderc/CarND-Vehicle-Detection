**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image13.png
[image2]: ./output_images/image0123.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 77 through 79 of the file called `trainSVC.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image2] ![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

Initially I tried some of the parameters suggested in the Q&A video, as well as parameters that I felt gave the highest classifier accuracy on a 1000 random image selection from the data. This meant using the LUV colorspace, 10 orientations, 10 cells per block, and all channels for HOGs. This was in part since I was low on time to pass this project. However, when I started having difficulties in getting good detections without false positives, I started to try out various combinations of parameters and eventually settled on using the YCrCb colorspace, which, even though it had lower test set accuracy, performed better in the video. I ended up adjusting the other parameters a bit more to make sure that the detections in the video seemed ok.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In lines 83 to 105, I trained a linear SVM classifier. I stacked HOG, spatial, and histogram features into a vector and scaled them using the standard scaler provided by sklearn. I saved the scaler to a file using the joblib dump function, so that I could later scale other images that I would predict. I then split the data into train and test sets using a random seed, and then trained the classifier using the fit function provided by sklearn. Once trained, the model was tested on the test data and then saved into pickle file using the joblib dump function, so that it could later also be using for predictions.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I initially set up a single size of sliding window with a scale of 64 pixels and decided that 50% overlap was the most I should do. This made for a very fast detection algorithm, but with low detection rates, since it wasn't very robust. Eventually I tried a second scale, varying in size from 40 pixels all the way up to 192 pixels. When I found that the large windows did not improve detection much and were significantly larger than the cars, I decided on using three scales of 64, 96, and 128 pixels, searching along the lower half of the image. When I found the detection missed a white car for a moment, I increased the height of the image to search in, as the car got fairly close to the horizon height at one point in the video. Following is an example of the sliding windows:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features, plus spatially binned color, and histograms of color in the feature vector, which provided a fairly good result. However, there were still false positives and gaps in the detection of the vehicles, but it was clear that with this dataset it would be difficult to do much better. Eventually I used some averaged heatmaps to take care of this, explained later. Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4). There are a couple of moments where a vehicle detection is lost and regained, or where an empty space is detected as a car, but these are fairly momentary. I think detection could be improved by augmenting the dataset, by using more different features, or even by applying a deep learning classifier to take advantage of its automatic feature selecting characteristic, which would make it much easier to see what might be relevant in distinguishing a car from a non-car, especially by using feature maps to see how a neural network would perform these detections.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. In order to make the detection more robust, I added up the heatmaps between frames and thresholded the sum of the heatmaps. This way, if a single frame had a false positive, but the next frame did not, that false positive would be easily swept away by the threshold, while a real car detection would increase its heatmap value and escape the threshold filter. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I faced hardware problems, when my code was causing the process to explode in memory consumption, causing my computer to crash as it attempted to swap into disk as fast as memory consumption increased. Once that was resolved, there were various moments when some obscure seeming errors would show up during execution, but I was able to trace down what was going on using print statements and tracking down other users' similar error messages on the web. However, I did feel that the classifier for this project was somewhat ineffective, since there were times when it couldn't tell the difference between shaded road or roadside vegetation and a car, and this led to some frustration. In addition, it was difficult to get a grasp of what colorspaces were more appropriate for use in detections, other than observing which ones provided better test set accuracies. I feel that my lack of experience with Python ended up hindering me some, since there were times when I couldn't figure out what was going on during the execution of my code to help me resolve unwanted behavior. This ensured that I took longer to get a working detection going.

It is very likely that my pipeline will fail in different lighting conditions, or on roads where different kinds of cars than those in the dataset might operate. Roads in a different country would certainly cause problems for my pipeline, since the cars would vary significantly from those in the dataset.

For this project I only used the datasets provided in the project and not the Udacity annotated dataset. I believe that increasing the size of the dataset and choosing images from a variety of different sources and situations would significantly increase the robustness of the pipeline. In addition, I strongly feel that the SVM classifier used in this project is not sufficiently adequate for the task, and that a properly structured and trained deep neural network architecture would be able to pick out better features for very robust and rapid detection, especially if measures are taken during training to make sure that the network generalizes well. However, if the SVM approach is preferable for some reason, some ways of making it more robust would be to search for cars in a small regions of the image. These would include near the last detections of the cars, near the horizon and near the lower sides of the images, which would make sure that any cars that enter the field of view would be tracked. If a detection should be lost in the middle of the expected region, then the pipeline could do a more thorough search to reacquire the detection. 
