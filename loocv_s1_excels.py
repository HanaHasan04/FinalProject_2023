import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import openpyxl

# Load pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Remove the last fully connected layer to obtain the feature extractor
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])

# Set the model to evaluation mode
resnet50.eval()


def encode_image(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)

    # Add a batch dimension to the image tensor
    image = image.unsqueeze(0)

    # Pass the image through the ResNet50 model to get the feature map
    feature_map = resnet50(image)

    # Detach the tensor from the computation graph before calling numpy()
    feature_map = feature_map.detach().numpy()

    return feature_map


# Define folders and classes
data_path = r"C:\Users\USER\Documents\UniversityProjects\PythonProjects\FinalProject\loocv_splits\S1"
folders = ["train", "test", "val"]
classes = ["Anticipation", "Baseline", "Disappointment", "Frustration"]

# Loop through each folder
for folder in folders:
    folder_path = os.path.join(data_path, folder)
    excel_file_path = os.path.join(data_path, folder + ".xlsx")
    df = pd.DataFrame(columns=["Image Name", "LABEL", "Vector"])

    # Loop through each class
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image_path = os.path.join(class_path, image_name)
                image_name_without_extension = os.path.splitext(image_name)[0]
                label = class_name
                vector = encode_image(image_path)
                df = pd.concat([df, pd.DataFrame(
                    {"Image Name": [image_name_without_extension], "LABEL": [label], "Vector": [vector]})],
                               ignore_index=True)
            else:
                print("UNIQUE EXTENSION")
                print(image_name)

    # Save DataFrame to Excel
    df.to_excel(excel_file_path, index=False)
    print(f"Excel file '{excel_file_path}' created successfully for folder '{folder}'")
