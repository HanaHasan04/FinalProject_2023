import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms


def embedding(frames_dir, metadata_dir, class_labels, i_start, i_end):
    # embedding model
    model = torch.hub.load('facebookresearch/dino:main', "dino_vitb8")
    embed_dim = model.embed_dim     # 768
    model.eval()

    # preprocessing before feeding an image to a ViT model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for i in range(i_start, i_end+1):
        data_path = os.path.join(frames_dir, f"S{i}")

        metadata_path = os.path.join(metadata_dir, f"metadata_{i}.xlsx")

        # define the columns of the Excel file
        column_names = ["Emotion", "Image", "Video", "Horse", "Frame"]
        feature_columns = [f'Feature_{j + 1}' for j in range(0, embed_dim)]
        all_columns = column_names + feature_columns
        df = pd.DataFrame(columns=all_columns)

        # loop through each emotion and process the images
        for class_name in class_labels:
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
                    # extract features from the image
                    with torch.no_grad():
                        features = model(processed_image)

                    # define the Excel row for the image
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

        df.to_excel(metadata_path, index=False)
        print(f"Embedding iteration {i}: Files saved successfully.")
