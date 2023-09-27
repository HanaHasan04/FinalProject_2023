from PIL import Image
import os


from PIL import Image
import os

def convert_frames(horse_path, frame_step=20):
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
                    num_frames = len(frames)
                    for i in range(0, num_frames):
                        if os.path.exists(frames[i]):
                            if i + 2 * frame_step < num_frames:
                                frame1_path = frames[i]
                                frame2_path = frames[i + frame_step]
                                frame3_path = frames[i + 2 * frame_step]
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


# Directory containing the frames
frames_directory = r"C:\Users\USER\Documents\UniversityProjects\FinalProject\AllDataYOLO"

start_i = 1
end_i = 30

for i in range(start_i, end_i+1):
    horse_dir = f"S{i}"
    print(horse_dir)
    horse_path = os.path.join(frames_directory, horse_dir)
    if os.path.isdir(horse_path):
        convert_frames(horse_path)

