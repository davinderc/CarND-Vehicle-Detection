from sklearn.externals import joblib
from slidingWindow import *
from search_wins import *
from scipy.ndimage.measurements import label
import time

def det_pipeline(img):

    ## Standard scaler, parameters, and SVC model
    x_scaler = joblib.load('x_scaler_save1.pkl')
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
    svc = joblib.load('svc_model1.pkl')


    t1 = time.time()
    draw_img = np.copy(img)

    windows = []
    l_windows = slide_window(img, x_start_stop=[0, img.shape[1]], y_start_stop=[360, 600], xy_window=(80, 80), xy_overlap=(0.5, 0.5))
    s_windows = slide_window(img, x_start_stop=[0, img.shape[1]], y_start_stop=[344, 536], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    windows.extend(l_windows)
    windows.extend(s_windows)

    hot_wins = search_windows(img, windows, svc, x_scaler, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel=hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    heat_map = add_heat(heat,hot_wins)

    labels = label(heat_map)

    draw_img = draw_labeled_bboxes(np.copy(img),labels)

    window_img = draw_boxes(draw_img, hot_wins, color=(0,0,255), thick=6)

    print(round(time.time() - t1,2),' seconds per image. Searching ', len(windows), ' windows.')

    return draw_img

image = 'test'
for i in range(6):
    j = 'test_images/test'+str(i+1)+'.jpg'
    k = 'test_images/test_det'+str(i+1)+'.jpg'
    l = 'test_images/test_det_labeled'+str(i+1)+'.jpg'
    img = cv2.imread(j)
    result = det_pipeline(img)
    cv2.imwrite(k,result)
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
