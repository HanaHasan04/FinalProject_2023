"""
Save 3 Excel files: train, val and test.

Each Excel files contains of the following columns:
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


# folders and classes.
data_path = r"C:\Users\USER\Documents\UniversityProjects\FinalProject\loocv_splits\S1"
folders = ["train", "test", "val"]
classes = ["Anticipation", "Baseline", "Disappointment", "Frustration"]

# embedding model.
model = torch.hub.load('facebookresearch/dino:main', "dino_vitb8")
embed_dim = model.embed_dim     # 768
model.eval()

# preprocessing before feeding an image to a ViT model.
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Resize((224, 224))
])

# create an Excel file for each folder, where folders = ["train", "test", "val"].
for folder in folders:
    folder_path = os.path.join(data_path, folder)
    excel_file_path = os.path.join(data_path, folder + ".xlsx")

    # define the columns of the Excel files.
    column_names = ["Emotion", "Image", "Video", "Horse", "Frame"]
    feature_columns = [f'Feature_{i + 1}' for i in range(0, embed_dim)]
    all_columns = column_names + feature_columns
    df = pd.DataFrame(columns=all_columns)

    # loop through each emotion.
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
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
                    **{f'Feature_{i + 1}': [features[0][i].item()] for i in range(embed_dim)}
                })], ignore_index=True)

            else:
                print("illegal image extension!")
                print(image_name)

    # save Excel file.
    df.to_excel(excel_file_path, index=False)
    print(f"Excel file '{excel_file_path}' created successfully.")
