import os
from pathlib import Path
import cv2

# TODO: balance dataset - total number of images of a horse should be the same for all horses.
# TODO: leave one object (horse) out - all images of the horse.

base_path = Path('C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject')
images_path = Path(os.path.join(base_path, 'Cropped_Resized')).glob('*')
split_data_path = os.path.join(base_path, 'data_split')

for images_folder in images_path:
    emotion = os.path.basename(images_folder)
    for image in os.listdir(images_folder):
        image_name = os.path.splitext(image)[0]
        im_path = os.path.join(os.path.join(os.path.join(os.path.join(base_path, "Cropped_Resized"), emotion)),
                               str(image_name) + ".jpg")
        # Load the image
        img = cv2.imread(im_path)

        subject_num = image_name[1:3]
        if subject_num[1] == '-':
            subject_num = subject_num[0]
        subject_num = int(subject_num)

        if subject_num>=25 and subject_num<=27:
            file_name = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(base_path, "data_split"), "val"), emotion)),
                               str(image_name) + '.jpg')
            cv2.imwrite(file_name, img)
        elif subject_num>=28 and subject_num<=30:
            file_name = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(base_path, "data_split"), "test"), emotion)),
                               str(image_name) + '.jpg')
            cv2.imwrite(file_name, img)
        else:
            file_name = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(base_path, "data_split"), "train"), emotion)),
                               str(image_name) + '.jpg')
            cv2.imwrite(file_name, img)