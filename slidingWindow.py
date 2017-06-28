import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lessonFunctions import *

#image = mpimg.imread('bbox-example-image.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if(x_start_stop == [None,None]):
        x_start_stop = [0,img.shape[1]]
    if(y_start_stop == [None,None]):
        y_start_stop = [np.int(0.6*img.shape[0]),img.shape[0]]

    # Compute the span of the region to be searched
    span_x = x_start_stop[1] - x_start_stop[0]
    span_y = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    step_x = np.int(xy_window[0]*(1 - xy_overlap[0]))
    step_y = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    x_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    y_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    num_win_x = np.int((span_x - x_buffer)/step_x)
    num_win_y = np.int((span_y - y_buffer)/step_y)

    # = span_x//step_x
    # = span_y//step_y

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for i in range(num_win_x):
        for j in range(num_win_y):
            startx = i*step_x + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = j*step_y + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    return window_list

# windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
#                     xy_window=(128, 128), xy_overlap=(0.5, 0.5))
#
# features = []
# for window in windows:
# 	feature = extract_features(image[window])
#
# window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
# plt.imshow(window_img)
