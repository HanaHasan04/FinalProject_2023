import os
import cv2


def extract_and_save_video_frames(videos_dir, frames_dir):
    for emotion in os.listdir(videos_dir):
        for video in os.listdir(os.path.join(videos_dir, emotion)):
            video_name, _ = os.path.splitext(video)
            horse_number = int(str(video_name).split("-")[0][1:])
            horse_path = os.path.join(frames_dir, 'S' + str(horse_number))
            emotion_path = os.path.join(horse_path, emotion)

            vid = os.path.join(videos_dir, emotion, f'{video_name}.mp4')
            cap = cv2.VideoCapture(vid)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                img_name = os.path.join(emotion_path, f'{video_name}__{i}.jpg')
                print(img_name)
                cv2.imwrite(img_name, frame)
                i += 1
            cap.release()
    cv2.destroyAllWindows()
