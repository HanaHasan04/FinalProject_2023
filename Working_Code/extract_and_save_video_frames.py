```python
import os
from pathlib import Path
import cv2

def extract_and_save_video_frames():
    path = Path('C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject')
    videos_path = Path(os.path.join(path, 'Videos')).glob('*')
    for video_folder in videos_path:
        emotion = os.path.basename(video_folder)
        print(emotion)
        images_path = os.path.join(path, 'Images')
        print(images_path)
        for video in os.listdir(video_folder):
            video_name = os.path.splitext(video)[0]
            tmp = os.path.join(os.path.join(os.path.join(os.path.join(path, "Videos"), emotion)), str(video_name)+".mp4")
            print(tmp)
            cap = cv2.VideoCapture(tmp)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                img_name = os.path.join(os.path.join(os.path.join(os.path.join(path, "Images"), emotion)),
                                   str(video_name) + '__' + str(i) + '.jpg')
                print(img_name)
                cv2.imwrite(img_name, frame)
                i += 1
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    extract_and_save_video_frames()

```
