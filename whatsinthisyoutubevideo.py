# Import statements ===========================================================

# Suggested import statement for youtube-dl
from __future__ import unicode_literals
# youtube-dl (https://github.com/ytdl-org/youtube-dl)
import youtube_dl

# opencv-python (https://github.com/opencv/opencv-python)
import cv2

# Used for yolov5 (https://github.com/ultralytics/yolov5)
# Also check out https://github.com/ultralytics/yolov5/issues/36 for further
# explanation
import torch

# matplotlib (https://github.com/matplotlib/matplotlib)
import matplotlib
import matplotlib.pyplot as plt

# Standard Python libraries
import argparse
import os
import shutil
import contextlib

# =============================================================================


# Code, with some hopefully helpful comments ==================================

parser = argparse.ArgumentParser()
parser.add_argument("youtube_url",
    help="full url for a youtube video, like "
    "https://www.youtube.com/watch?v=69V__a49xtw")
parser.add_argument("--colab", help="for running in Google Colab",
    action="store_true")
args = parser.parse_args()
youtube_url = args.youtube_url
colab = args.colab

# Change this variable to change how many frames per second of the YouTube
# video will be analyzed by the computer vision model
how_many_frames_per_second = 6

# All files created in this program will be in temp directory
# (The temp directory gets deleted at the end of the program)
temp_dir = os.path.join(os.path.dirname(__file__), "temp")
temp_video_dir = os.path.join(temp_dir, "video")
temp_images_dir = os.path.join(temp_dir, "images")
os.mkdir(temp_dir)

# Download video with youtube-dl ----------------------------------------------
os.mkdir(temp_video_dir)

output_template = os.path.join(temp_video_dir, "video-%(duration)s.%(ext)s")

ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'outtmpl': output_template,
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([youtube_url])
# Video is now saved in temp directory
# -----------------------------------------------------------------------------

for filename in os.listdir(temp_video_dir):
    video_filename = filename
video_duration_sec = int(((video_filename.split("-"))[1].split("."))[0])

# Extract images/frames with opencv-python (cv2) ------------------------------
os.mkdir(temp_images_dir)

image_count = 1

vidcap = cv2.VideoCapture(os.path.join(temp_video_dir, video_filename))

video_fps = round(vidcap.get(cv2.CAP_PROP_FPS))
frames_to_skip = video_fps // how_many_frames_per_second
frame_count = 0

success, image = vidcap.read()
while success:
    if frame_count == 0:
        cv2.imwrite(os.path.join(temp_images_dir,
            f"image-{image_count}.jpg"), image)
        image_count += 1
    frame_count += 1
    frame_count %= frames_to_skip
    success, image = vidcap.read()

vidcap.release()
cv2.destroyAllWindows()
# -----------------------------------------------------------------------------

num_frames = image_count - 1

# Run yolov5 on the images ----------------------------------------------------
with open(os.path.join(temp_dir, "trash"), 'w') as f:
    with contextlib.redirect_stdout(f):
        with contextlib.redirect_stderr(f):
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            model.conf = 0.7

images = []
for filename in os.listdir(temp_images_dir):
    images.append(os.path.join(temp_images_dir, filename))

results = model(images, size=320)

classes_dict = {}

panda = results.pandas().xyxy

for i in range(num_frames):
    classes = set(panda[i].name)
    for c in classes:
        if c not in classes_dict:
            classes_dict[c] = 1
        else:
            classes_dict[c] += 1
# -----------------------------------------------------------------------------

classes_sorted = sorted(classes_dict.items(), key=lambda item: item[1],
    reverse=True)
class_names = []
class_num_frames = []

print(f"YouTube video was {video_duration_sec} seconds long")
print(f"For every second of video, roughly {how_many_frames_per_second} "
    f"frames were analyzed (for a total of {num_frames} frames)\n")
print("Here's what's in this YouTube video:")
for c in classes_sorted:
    print(f"A {c[0]} in {c[1]} of {num_frames} frames or "
        f"{round(100*c[1]/num_frames, 2)}% of the video")
    class_names.append(c[0].capitalize())
    class_num_frames.append(round(100*c[1]/num_frames, 2))

# Plot some stuff with matplotlib ---------------------------------------------
if not colab:
    matplotlib.use("TkAgg")
plt.barh(class_names[::-1], class_num_frames[::-1], align='center')
plt.xlabel("How much it's in the video, in percent")
plt.xlim([0, 100])
plt.title("What's in this YouTube video?")
for i, v in enumerate(class_num_frames[::-1]):
    plt.text(v, i, str(v))
plt.show()
if colab:
    plt.savefig('figure.jpg')
# -----------------------------------------------------------------------------

# Remove/delete the temp directory
shutil.rmtree(temp_dir)

# =============================================================================
