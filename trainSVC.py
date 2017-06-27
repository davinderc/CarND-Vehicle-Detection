#from carNotCar import *
import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lessonFunctions import *

v_dir = "vehicles/"
nv_dir = "non-vehicles/"

v_subdir = os.listdir(v_dir)
nv_subdir = os.listdir(nv_dir)

cars = []
notcars = []

for subdir in v_subdir:
    cars.extend(glob.glob(v_dir + subdir + '/*'))
print('Number of Vehicle Images found: ', len(cars))
with open("cars.txt", 'w') as f:
    for fn in cars:
        f.write(fn+'\n')

for subdir in nv_subdir:
    notcars.extend(glob.glob(nv_dir + subdir + '/*'))
print('Number of Non-Vehicle Images found: ', len(notcars))
with open("notcars.txt", 'w') as f:
    for fn in notcars:
        f.write(fn+'\n')

car_i = np.random.randint(0,len(cars))
notcar_i = np.random.randint(0,len(cars))

#car_image = mpimg.imread(cars[car_i])
#notcar_image = mpimg.imread(notcars[notcar_i])

#print(car_image.shape)
#print(type(car_image[0][0][0]))
#feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2LUV)

color_space = 'YUV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32,32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

#car_features, car_hog_image = get_hog_features(c_gray, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
#notcar_features, notcar_hog_image = get_hog_features(nc_gray, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)


t = time.time()
n_samples = 1000
random_i = np.random.randint(0,len(cars),n_samples)
test_cars = np.array(cars)[random_i]
test_notcars = np.array(notcars)[random_i]

car_features = extract_features(test_cars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

notcar_features = extract_features(test_notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

print(time.time()-t, 'seconds to compute features' )

#images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
titles = ['car', 'car HOG', 'notcar', 'notcar HOG']

#fig = plt.figure(figsize=(12,3))
# plt.ion()
# plt.imshow(images[0])
# plt.show()
# plt.waitforbuttonpress()
# plt.imshow(images[1],cmap='hot')
# plt.show()
# plt.waitforbuttonpress()
# plt.imshow(images[2])
# plt.show()
# plt.waitforbuttonpress()
# plt.imshow(images[3],cmap='hot')
# plt.show()
# plt.waitforbuttonpress()
