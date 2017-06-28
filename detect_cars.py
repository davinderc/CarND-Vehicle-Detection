def detect_car(img, svc):
    windows = slide_window(img)
    to_draw = []
    for window in windows:
        if(svc.predict(img[window])):
            to_draw.append(window)
    detected = draw_boxes(img,to_draw)
    return detected
