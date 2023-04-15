import os
from pathlib import Path
import cv2

base_path = Path('C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject')
images_path = Path(os.path.join(base_path, 'Images')).glob('*')
cropped_resized_path = os.path.join(base_path, 'Cropped_Resized')

for images_folder in images_path:
    emotion = os.path.basename(images_folder)
    for image in os.listdir(images_folder):
        image_name = os.path.splitext(image)[0]
        im_path = os.path.join(os.path.join(os.path.join(os.path.join(base_path, "Images"), emotion)), str(image_name) + ".jpg")
        # Load the image
        img = cv2.imread(im_path)
        # Get the size of the image
        height, width = img.shape[:2]
        # Crop the image to a fixed size
        fixed_size = (1080, 1080)
        start_x = int(width / 2 - fixed_size[0] / 2)
        start_y = int(height / 2 - fixed_size[1] / 2)
        cropped_img = img[start_y:start_y + fixed_size[1], start_x:start_x + fixed_size[0]]
        # Resize the image to a fixed size
        fixed_size = (224, 224)
        resized_img = cv2.resize(cropped_img, fixed_size, interpolation=cv2.INTER_AREA)
        file_name = os.path.join(os.path.join(os.path.join(os.path.join(base_path, "Cropped_Resized"), emotion)),
                           str(image_name) + '.jpg')
        cv2.imwrite(file_name, resized_img)
