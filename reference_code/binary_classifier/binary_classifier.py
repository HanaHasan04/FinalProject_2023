import os


def binary_classifier(frames_dir, two_emotions_frames_dir, i_start, i_end):
    for i in range(i_start, i_end + 1):
        horse_dir = os.path.join(frames_dir, f"S{i}")

        for emotion_dir in os.listdir(horse_dir):
            if emotion_dir != "Anticipation" and emotion_dir != "Frustration":
                emotion_path = os.path.join(horse_dir, emotion_dir)

                for frame in os.listdir(emotion_path):
                    if frame.endswith('.jpg') or frame.endswith('.jpeg'):
                        frame_path = os.path.join(emotion_path, frame)
                        os.remove(frame_path)
