import os
import random
import shutil
from pathlib import Path

base_path = Path('C:/Users/USER/Documents/UniversityProjects/FinalProject')
data_path = Path(os.path.join(base_path, 'Images')).glob('*')
loocv_path = os.path.join(base_path, 'loocv_splits')


def loocv_train_test_split(horse):
    loocv_folder = os.path.join(loocv_path, horse)

    # the horse number
    no = int(horse[1:])
    r = list(range(1, no)) + list(range(no+1, 31))
    val_horses_numbers = random.sample(r, 3)

    for folder in data_path:
        emotion = os.path.basename(folder)
        for image in os.listdir(folder):
            image_name = os.path.splitext(image)[0]
            horse_name = image_name.split("-")[0]
            horse_number = int(horse_name[1:])
            image_path = os.path.join(os.path.join(os.path.join(os.path.join(base_path, "Images"), emotion)),
                                   str(image_name) + ".jpg")
            # image_path = os.path.join(data_path, emotion, f"{image_name}.jpg")

            # test
            if no == horse_number:
                output_path = os.path.join(loocv_folder, "test", emotion)
                shutil.copy(image_path, output_path)

            # val
            elif horse_number in val_horses_numbers:
                output_path = os.path.join(loocv_folder, "val", emotion)
                shutil.copy(image_path, output_path)

            # train
            else:
                output_path = os.path.join(loocv_folder, "train", emotion)
                shutil.copy(image_path, output_path)

n_horses = 30
for i in range(1, n_horses):
  loocv_train_test_split("S" + str(i))
