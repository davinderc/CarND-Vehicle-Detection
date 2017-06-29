from sklearn.externals import joblib
from slidingWindow import *
from search_wins import *
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import collections

heatmaps = collections.deque(maxlen=14)

#vehicles96 = VehicleTrack((96,86))
#vehicles64 = VehicleTrack((64,64))

def det_pipeline(img,i):
    #print(img.shape)
    ## Standard scaler, parameters, and SVC model
    x_scaler = joblib.load('x_scaler_save5.pkl')
    color_space = 'YCrCb'
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    spatial_size = (16,16)
    hist_bins = 16
    spatial_feat = True
    hist_feat = True
    hog_feat = True
    svc = joblib.load('svc_model5.pkl')

    t1 = time.time()
    draw_img = np.copy(img)

    windows = []
    l_windows = slide_window(img, x_start_stop=[0, img.shape[1]], y_start_stop=[316, 664], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    m_windows = slide_window(img, x_start_stop=[0, img.shape[1]], y_start_stop=[304, 640], xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    #s_windows = slide_window(img, x_start_stop=[0, img.shape[1]], y_start_stop=[280, 600], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    windows.extend(l_windows)
    windows.extend(m_windows)
    #windows.extend(s_windows)
    #windowed_img = draw_boxes(draw_img, l_windows, color=(255,0,0), thick=6)


    hot_wins = search_windows(img, windows, svc, x_scaler, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel=hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True)
    searched1 = draw_boxes(draw_img, hot_wins, color=(0,0,255), thick=6)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    heat_map = add_heat(heat,hot_wins)
    heatmaps.append(heat_map)
    heatmap_sum = sum(heatmaps)
   #print('heatmap: ', np.max(heatmap_sum))
    thresh_heat = apply_threshold(heatmap_sum, 6)
    labels = label(thresh_heat)
    # if(i == 6):
    #     draw_img = draw_labeled_bboxes(np.copy(img),labels)
    #     plt.imsave('./six_frames/resulting.png',draw_img)
    #
    # return heat_map
    # import sys
    # sys.exit()

    draw_img = draw_labeled_bboxes(np.copy(img),labels)
    #window_img = draw_boxes(draw_img, labels, color=(0,0,255), thick=6)

    #print(round(time.time() - t1,2),' seconds per image. Searching ', len(windows), ' windows.')

    return draw_img

import os
frames = os.listdir('./six_frames')
i = 1
for frame in frames:
    img = mpimg.imread('./six_frames/'+frame)
    heat = det_pipeline(img,i)
    i +=1
    print(np.max(heat))
    plt.imsave('./six_frames/heat'+frame,heat,cmap='hot')

# image = 'test'
# for i in range(6):
#     j = 'test_images/test'+str(i+1)+'.jpg'
#     k = 'test_images/test_det'+str(i+1)+'.jpg'
#     l = 'test_images/test_det_labeled'+str(i+1)+'.jpg'
#     img = cv2.imread(j)
#     result = det_pipeline(img)
#     cv2.imwrite(k,result)
    #cv2.imwrite(l,labeled)
    ## Select windows from image, extract features and draw boxes on detections
    # windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    # car_windows = []
    # for coords in windows:
    #     window = img[coords[0][0]:coords[1][0],coords[0][1]:coords[1][1]]
    #     features = extract_features(window, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
    #
    #     scaled_feat = x_scaler.transform(features)
    #     if(svc.predict(scaled_feat)):
    #         car_windows.append(coords)
    # if(car_windows is not None):
    #     draw_boxes(img,car_windows)
    #     cv2.imwrite(img,'test_detect.jpg')
