import os
from PIL import Image
import os
from pathlib import Path
import cv2
import torch
from torch import IntTensor


def process_frames(directory):
    for emotion_dir in os.listdir(directory):
        emotion_path = os.path.join(directory, emotion_dir)
        if os.path.isdir(emotion_path):
            for filename in os.listdir(emotion_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    frame_path = os.path.join(emotion_path, filename)
                    if False:
                        os.remove(frame_path)
                    else:
                        frame = cv2.imread(frame_path)

                        height, width, _ = frame.shape

                        if frame is not None:
                            if height != 224 or width != 224:
                                results = model(frame)
                                v = results.xyxy[0]
                                num_of_objects = v.shape[0]
                                horse_detected = False
                                for i in range(num_of_objects):
                                    if v[i][5] == YOLO_horse:
                                        xmin = int(IntTensor.item(v[i][0]))
                                        ymin = int(IntTensor.item(v[i][1]))
                                        xmax = int(IntTensor.item(v[i][2]))
                                        ymax = int(IntTensor.item(v[i][3]))
                                        frame = frame[ymin:ymax, xmin:xmax]
                                        frame = cv2.resize(frame, (224, 224))
                                        horse_detected = True
                                        break  # No need to continue iterating if a horse is detected

                                if horse_detected:
                                    cv2.imwrite(frame_path, frame)
                                else: # No horse detected - do center crop.
                                    height, width = frame.shape[:2]
                                    target_size = min(height, width)
                                    top = (height - target_size) // 2
                                    left = (width - target_size) // 2
                                    bottom = top + target_size
                                    right = left + target_size
                                    frame = frame[top:bottom, left:right]
                                    frame = cv2.resize(frame, (224, 224))
                                    cv2.imwrite(frame_path, frame)


def convert_frames(directory):
    for horse_dir in os.listdir(directory):
        horse_path = os.path.join(directory, horse_dir)
        if os.path.isdir(horse_path):
            for emotion_dir in os.listdir(horse_path):
                emotion_path = os.path.join(horse_path, emotion_dir)
                if os.path.isdir(emotion_path):
                    video_frames = {}  # Dictionary to store frames for each video
                    frame_files = sorted([f for f in os.listdir(emotion_path) if f.endswith('.jpg') or f.endswith('.jpeg')])
                    frame_files.sort(key=lambda x: int(x.split('__')[-1].split('.')[0]))

                    for frame_file in frame_files:
                        frame_path = os.path.join(emotion_path, frame_file)
                        video_name = frame_file.split('__')[0]
                        if video_name not in video_frames:
                            video_frames[video_name] = []
                        video_frames[video_name].append(frame_path)

                    for video_name, frames in video_frames.items():
                        j = -1
                        num_frames = len(frames)
                        for i in range(0, num_frames, 3):
                            if i + 2 < num_frames:
                                j += 1
                                frame1_path = frames[i]
                                frame2_path = frames[i + 1]
                                frame3_path = frames[i + 2]
                                frame1 = Image.open(frame1_path).convert('L')
                                frame2 = Image.open(frame2_path).convert('L')
                                frame3 = Image.open(frame3_path).convert('L')
                                new_image = Image.merge("RGB", (frame1, frame2, frame3))
                                os.remove(frame1_path)
                                os.remove(frame2_path)
                                os.remove(frame3_path)
                                new_image_filename = os.path.basename(frame1_path)
                                new_image_path = os.path.join(emotion_path, new_image_filename)
                                new_image.save(new_image_path)

                            elif i + 1 < num_frames:
                                frame1_path = frames[i]
                                frame2_path = frames[i + 1]
                                os.remove(frame1_path)
                                os.remove(frame2_path)
                            else:
                                frame1_path = frames[i]
                                os.remove(frame1_path)


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

YOLO_horse = 17

# Directory containing the frames
frames_directory = r"C:\Users\USER\Documents\UniversityProjects\FinalProject\AllDataYOLO"

start_i = 16
end_i = 30

for i in range(start_i, end_i+1):
    horse_dir = f"S{i}"
    print(horse_dir)
    horse_path = os.path.join(frames_directory, horse_dir)
    if os.path.isdir(horse_path):
        process_frames(horse_path)


# # Step 2: Convert consecutive frames to new RGB images
# convert_frames(frames_directory)
