from moviepy.editor import VideoFileClip
from detection import det_pipeline

test_output = 'test_full4.mp4'

clip = VideoFileClip('project_video.mp4')

test_clip = clip.fl_image(det_pipeline)
test_clip.write_videofile(test_output, audio=False)
