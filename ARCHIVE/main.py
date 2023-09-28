import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filepath = "C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject/Images/Anticipation/S1-T1-A1-C1-3__0.jpg"
    image = cv2.imread(filepath)
    im = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    height = im.shape[0]
    width = im.shape[1]
    print(height)
    print(width)
    cv2.imshow("HI", im)
    cv2.waitKey(0)

    # extract_and_save_video_frames()
    # cv2.imwrite("Hi, LEON!", os.path.join("C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\pyimg\\images", "OLA.jpg"))
    # cap = cv2.VideoCapture("C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\FinalProject\\Videos\\Baseline\\S1-B-3.mp4")
    # im = cv2.imread('C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\pyimg\\images\\StarWars_wallpaper.png')
    # cv2.imwrite("C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\pyimg\\images\\HELLO.jpg", im)




