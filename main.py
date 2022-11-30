import os
from pathlib import Path
import cv2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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
    extract_and_save_video_frames()
    # cv2.imwrite("Hi, LEON!", os.path.join("C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\pyimg\\images", "OLA.jpg"))
    # cap = cv2.VideoCapture("C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\FinalProject\\Videos\\Baseline\\S1-B-3.mp4")
    # im = cv2.imread('C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\pyimg\\images\\StarWars_wallpaper.png')
    # cv2.imwrite("C:\\Users\\USER\\Documents\\UniversityProjects\\PythonProjects\\pyimg\\images\\HELLO.jpg", im)
