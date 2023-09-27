import os
from PIL import Image


def convert_frames_to_GrayST(frames_dir, i_start, i_end, frame_step=20):
    for i in range(i_start, i_end + 1):
        horse_dir = os.path.join(frames_dir, f"S{i}")
        if os.path.isdir(horse_dir):
            for emotion_dir in os.listdir(horse_dir):
                emotion_path = os.path.join(horse_dir, emotion_dir)
                if os.path.isdir(emotion_path):
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
                                if j + 2 * frame_step < num_frames:
                                    frame1_path = frames[j]
                                    frame2_path = frames[j + frame_step]
                                    frame3_path = frames[j + 2 * frame_step]
                                    frame1 = Image.open(frame1_path).convert('L')
                                    frame2 = Image.open(frame2_path).convert('L')
                                    frame3 = Image.open(frame3_path).convert('L')
                                    new_image = Image.merge("RGB", (frame1, frame2, frame3))
                                    frame1_num = frame1_path.split('__')[-1].split('.')[0]
                                    frame2_num = frame2_path.split('__')[-1].split('.')[0]
                                    frame3_num = frame3_path.split('__')[-1].split('.')[0]
                                    frame_nums = f"{frame1_num}_{frame2_num}_{frame3_num}"
                                    new_image_filename = f"{video_name}__{frame_nums}.jpg"
                                    new_image_path = os.path.join(emotion_path, new_image_filename)
                                    new_image.save(new_image_path)

                                    # Remove the processed frames
                                    os.remove(frame1_path)
                                    os.remove(frame2_path)
                                    os.remove(frame3_path)
        print(f"GrayST iteration {i}: Files saved successfully.")
