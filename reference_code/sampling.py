"""
    Change:
        - In main.py:
            call the functions in this order:
            1. extract_and_save_video_frames(videos_dir, frames_dir)
            2. sampling(frames_dir, i_start, i_end, step)
            3. yolo_detction(frames_dir, i_start, i_end)

    Use the original code files:
        - extract_frames.py
        - yolo.py
        - grayST.py
        - embedding.py
        - train.py
        - average.py
"""

import os


def sampling(frames_dir, i_start, i_end, step=5):
    for i in range(i_start, i_end + 1):
        horse_dir = os.path.join(frames_dir, f"S{i}")

        for emotion_dir in os.listdir(horse_dir):
            emotion_path = os.path.join(horse_dir, emotion_dir)

            video_frames = {}  # dictionary to store frames for each video
            frame_files = sorted(
                [f for f in os.listdir(emotion_path) if f.endswith('.jpg') or f.endswith('.jpeg')])
            frame_files.sort(key=lambda x: int(x.split('__')[-1].split('.')[0]))

            for frame_file in frame_files:
                frame_path = os.path.join(emotion_path, frame_file)
                video_name = frame_file.split('__')[0]
                if video_name not in video_frames:
                    video_frames[video_name] = []
                video_frames[video_name].append(frame_path)

            for video_name, frames in video_frames.items():
                num_frames = len(frames)
                for j in range(0, num_frames):
                    if os.path.exists(frames[j]):
                        if j % step != 0:
                            os.remove(frames[j])
