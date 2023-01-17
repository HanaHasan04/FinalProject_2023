import os
from pathlib import Path
import cv2
import torch
from torch import IntTensor

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

base_path = Path('C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject')
images_path = Path(os.path.join(base_path, 'Images')).glob('*')
raw_data_path = os.path.join(base_path, 'Raw_Data')

for images_folder in images_path:
    emotion = os.path.basename(images_folder)
    for image in os.listdir(images_folder):
        image_name = os.path.splitext(image)[0]
        im_path = os.path.join(os.path.join(os.path.join(os.path.join(base_path, "Images"), emotion)), str(image_name) + ".jpg")
        im = cv2.imread(im_path)
        # Inference
        results = model(im)
        # Format of the results: a table where each row is an object,
        # columns: xmin, ymin, xmax, ymax, confidence, class (coordinates of bounding box, confidence score, object label)
        v = results.xyxy[0]
        # Detecting horses only
        num_of_objects = v.shape[0]
        horse_class = 17
        for i in range(num_of_objects):
            if v[i][5] == horse_class:
                xmin = int(IntTensor.item(v[i][0]))
                ymin = int(IntTensor.item(v[i][1]))
                xmax = int(IntTensor.item(v[i][2]))
                ymax = int(IntTensor.item(v[i][3]))
                # crop the image around the horse's face
                crop_im = im[ymin:ymax, xmin:xmax]
                small_im = cv2.resize(crop_im, (0, 0), fx=0.5, fy=0.5)
                file_name = os.path.join(os.path.join(os.path.join(os.path.join(base_path, "Raw_Data"), emotion)),
                                   str(image_name) + '.jpg')
                cv2.imwrite(file_name, small_im)
