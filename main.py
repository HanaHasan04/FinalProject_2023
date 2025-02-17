"""
    The dataset:
        - In 'data/videos' there are 4 directories:
            1. 'data/videos/Anticipation' with ~3 videos (of 3 seconds) per horse.
            2. 'data/videos/Baseline' with ~1 video (of 3 seconds) per horse.
            3. 'data/videos/Disappointment' with ~3 videos (of 3 seconds) per horse.
            4. 'data/videos/Frustration' with ~3 videos (of 3 seconds) per horse.

        - Example of video's name: 'S1-T1-A1-C1-3.mp4'
            1. 'S1': horse number 1.
            2. 'T1-A1-C1-3': the name of the video.
"""


from extract_frames import *
from yolo import *
from grayST import *
from embedding import *
from train import *
from average import *

import time

videos_dir = 'data/videos'
frames_dir = 'data/frames'
os.makedirs(frames_dir, exist_ok=True)
metadata_dir = 'data/metadata'
os.makedirs(metadata_dir, exist_ok=True)
new_metadata_dir = 'data/new_metadata'
os.makedirs(new_metadata_dir, exist_ok=True)
metrics_dir = 'data/metrics'
os.makedirs(metrics_dir, exist_ok=True)
new_metrics_dir = 'data/new_metrics'
os.makedirs(new_metrics_dir, exist_ok=True)
excels_dir = 'data/excels'
os.makedirs(excels_dir, exist_ok=True)

excel_first_model = "yolo_grayST"
excel_second_model = "yolo_grayST_top_k"


i_start = 1
i_end = 30

# parameters
frame_step = 20
k = 100
k_baseline = 34

class_labels = ["Anticipation", "Baseline", "Disappointment", "Frustration"]


print("** VIDEO-BASED AUTOMATED RECOGNITION OF EMOTIONAL STATES IN HORSES **")
print("")

for i in range(i_start, i_end + 1):
    horse_folder = os.path.join(frames_dir, f'S{i}')
    os.makedirs(horse_folder, exist_ok=True)
    os.makedirs(os.path.join(horse_folder, 'Anticipation'), exist_ok=True)
    os.makedirs(os.path.join(horse_folder, 'Baseline'), exist_ok=True)
    os.makedirs(os.path.join(horse_folder, 'Disappointment'), exist_ok=True)
    os.makedirs(os.path.join(horse_folder, 'Frustration'), exist_ok=True)

print("STEP 1: EXTRACT FRAMES FROM VIDEOS")
start_time = time.time()
extract_and_save_video_frames(videos_dir, frames_dir)
print("Time: ", time.time() - start_time)
print("")

print("STEP 2: YOLO DETECTION")
start_time = time.time()
yolo_detction(frames_dir, i_start, i_end)
print("Time: ", time.time() - start_time)
print("")

print("STEP 3: GRAY-ST")
start_time = time.time()
convert_frames_to_GrayST(frames_dir, i_start, i_end, frame_step)
print("Time: ", time.time() - start_time)
print("")

print("STEP 4: EMBEDDING")
start_time = time.time()
embedding(frames_dir, metadata_dir, class_labels, i_start, i_end)
print("Time: ", time.time() - start_time)
print("")

retrain = False
print("STEP 5: TRAIN A FIRST MODEL USING ALL DATA")
start_time = time.time()
train_the_model(metadata_dir, new_metadata_dir, metrics_dir, new_metrics_dir, class_labels, k, k_baseline, i_start, i_end, retrain)
first_average = calculate_average_results(metrics_dir, excels_dir, excel_first_model, i_start, i_end)
print(f"Average accuracy for all horses: {first_average}")
print("Time: ", time.time() - start_time)
print("")

retrain = True
print(f"STEP 6: TRAIN A SECOND MODEL USING THE TOP K = {k} FRAMES")
start_time = time.time()
train_the_model(metadata_dir, new_metadata_dir, metrics_dir, new_metrics_dir, class_labels, k, k_baseline, i_start, i_end, retrain)
second_average = calculate_average_results(new_metrics_dir, excels_dir, excel_second_model, i_start, i_end)
print(f"Average accuracy for all horses: {second_average}")
print("Time: ", time.time() - start_time)
print("")