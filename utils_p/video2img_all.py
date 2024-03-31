import glob
from video2img import video2img


for video_path in sorted(glob.glob("data/pseudo_viz/*.mp4")):
    video2img(video_path)
