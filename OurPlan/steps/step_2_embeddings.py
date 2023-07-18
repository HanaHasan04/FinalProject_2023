"""
Save a single Excel file containing all the data.

The Excel file contains the following columns:
- Emotion: the ground-truth label of the image, in {"Anticipation", "Baseline", "Disappointment", "Frustration"}
- Image: image name (without extension)
- Video: video name (without extension)
- Horse: the number of the horse (1-30)
- Frame: the number of the frame, which is its "order" in the video (usually 0-180)
- Feature_{i+1}: i in [0, 768). the i-th feature obtained from the embedding model. (i depends on "embed_dim").
"""

import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import time

# Define the range of i values
i_start = 1
i_end = 30

# Define the base data path
base_data_path = "data/frames"
metadata_dir = "data/metadata"
os.makedirs(metadata_dir, exist_ok=True)

# embedding model.
model = torch.hub.load('facebookresearch/dino:main', "dino_vitb8")
embed_dim = model.embed_dim     # 768
model.eval()

# preprocessing before feeding an image to a ViT model.
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# loop through each i value
for i in range(i_start, i_end+1):
    t = time.time()
    # folder and classes.
    data_path = os.path.join(base_data_path, f"S{i}")
    classes = ["Anticipation", "Baseline", "Disappointment", "Frustration"]

    # create the Excel file.
    excel_file_path = os.path.join(metadata_dir, f"metadata_{i}.xlsx")

    # define the columns of the Excel file.
    column_names = ["Emotion", "Image", "Video", "Horse", "Frame"]
    feature_columns = [f'Feature_{j + 1}' for j in range(0, embed_dim)]
    all_columns = column_names + feature_columns
    df = pd.DataFrame(columns=all_columns)

    # loop through each emotion and process the images.
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        for image_name in os.listdir(class_path):
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image_path = os.path.join(class_path, image_name)
                image = Image.open(image_path)
                image_name = os.path.splitext(image_name)[0]
                video_name = image_name.split("__")[0]
                horse_name = image_name.split("-")[0]
                horse_number = int(horse_name[1:])
                frame_number = int(image_name.split("__")[1])
                label = class_name

                processed_image = torch.unsqueeze(preprocess(image), 0)
                # extract features from the image.
                with torch.no_grad():
                    features = model(processed_image)

                # define the Excel row for the image.
                df = pd.concat([df, pd.DataFrame({
                    "Emotion": [label],
                    "Image": [image_name],
                    "Video": [video_name],
                    "Horse": [horse_number],
                    "Frame": [frame_number],
                    **{f'Feature_{j + 1}': [features[0][j].item()] for j in range(embed_dim)}
                })], ignore_index=True)

            else:
                print("illegal image extension!")
                print(image_name)

    # save the Excel file.
    df.to_excel(excel_file_path, index=False)
    print(f"Excel file '{excel_file_path}' created successfully.")
    print("Time: ", time.time() - t)