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
from sklearn.externals import joblib
from lessonFunctions import *

v_dir = "vehicles/"
nv_dir = "non-vehicles/"

v_subdir = os.listdir(v_dir)
nv_subdir = os.listdir(nv_dir)

#print(v_subdir)
#print(nv_subdir)
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

#car_i = np.random.randint(0,len(cars))
#notcar_i = np.random.randint(0,len(cars))

#car_image = mpimg.imread(cars[car_i])
#notcar_image = mpimg.imread(notcars[notcar_i])

#print(car_image.shape)
#print(type(car_image[0][0][0]))
#feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2LUV)

color_space = 'LUV'
orient = 10
pix_per_cell = 10
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (24,24)
hist_bins = 24
spatial_feat = True
hist_feat = True
hog_feat = True

#car_features, car_hog_image = get_hog_features(c_gray, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
#notcar_features, notcar_hog_image = get_hog_features(nc_gray, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)


t = time.time()
n_samples_cars = 1000
n_samples_notcars = 1000
#n_samples_cars = len(cars)
#n_samples_notcars = len(notcars)
random_i_c = np.random.randint(0,len(cars),n_samples_cars)
random_i_nc = np.random.randint(0,len(cars),n_samples_notcars)
test_cars = np.array(cars)[random_i_c]
test_notcars = np.array(notcars)[random_i_nc]

car_features = extract_features(test_cars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

notcar_features = extract_features(test_notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

print(round(time.time() - t, 2), 'seconds to compute features' )

x = np.vstack((car_features, notcar_features)).astype(np.float64)

x_scaler = StandardScaler().fit(x)

joblib.dump(x_scaler,'x_scaler_save2.pkl')

scaled_x = x_scaler.transform(x)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

random_state = np.random.randint(0,100)

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.1, random_state = random_state)

print('Using ', orient, ' orientations, ', pix_per_cell, ' pixels per cell, ', cell_per_block, ' cells per block, ', hist_bins, ' histogram bins, and ', spatial_size, ' spatial sampling.')
print('Feature vector length: ', len(x_train[0]))

svc = LinearSVC()

t=time.time()
svc.fit(x_train,y_train)
print(round(time.time() - t, 2), ' seconds to train SVC...')
print('Test accuracy of SVC: ', round(svc.score(x_test, y_test),4))

joblib.dump(svc,'svc_model2.pkl')
#images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
#titles = ['car', 'car HOG', 'notcar', 'notcar HOG']

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
