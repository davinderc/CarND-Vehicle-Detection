from moviepy.editor import VideoFileClip
from detection import det_pipeline

test_output = 'test.mp4'

clip = VideoFileClip('test_video.mp4')

test_clip = clip.fl_image(det_pipeline)
test_clip.write_videofile(test_output, audio=False)
