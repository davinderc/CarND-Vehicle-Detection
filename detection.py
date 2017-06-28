from sklearn.externals import joblib
from slidingWindow import *
from search_wins import *

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

    draw_img = np.copy(img)

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    hot_wins = search_windows(img, windows, svc, x_scaler, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True)

    window_img = draw_boxes(draw_img, hot_wins, color=(0,0,255), thick=6)

    cv2.imwrite(window_img,'test_detect1.jpg')
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
