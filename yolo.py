import os
import cv2
from torch import IntTensor
import torch


def yolo_detction(frames_dir, i_start, i_end):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    YOLO_horse = 17

    for i in range(i_start, i_end + 1):
        horse_dir = os.path.join(frames_dir, f"S{i}")
        if os.path.isdir(horse_dir):
            for emotion_dir in os.listdir(horse_dir):
                emotion_path = os.path.join(horse_dir, emotion_dir)
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
                                        for j in range(num_of_objects):
                                            if v[j][5] == YOLO_horse:
                                                xmin = int(IntTensor.item(v[j][0]))
                                                ymin = int(IntTensor.item(v[j][1]))
                                                xmax = int(IntTensor.item(v[j][2]))
                                                ymax = int(IntTensor.item(v[j][3]))
                                                frame = frame[ymin:ymax, xmin:xmax]
                                                frame = cv2.resize(frame, (224, 224))
                                                horse_detected = True
                                                # No need to continue iterating if a horse is detected
                                                break

                                        if horse_detected:
                                            cv2.imwrite(frame_path, frame)
                                        else:
                                            # No horse detected - do center crop.
                                            height, width = frame.shape[:2]
                                            target_size = min(height, width)
                                            top = (height - target_size) // 2
                                            left = (width - target_size) // 2
                                            bottom = top + target_size
                                            right = left + target_size
                                            frame = frame[top:bottom, left:right]
                                            frame = cv2.resize(frame, (224, 224))
                                            cv2.imwrite(frame_path, frame)
        print(f"YOLO iteration {i}: Files saved successfully.")
