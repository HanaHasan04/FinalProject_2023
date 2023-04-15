import torch
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = models.resnet50(pretrained=True)

target_layer = model.layer4[-1]

img_path = 'path/to/your/image.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize((224, 224))
img = np.array(img)
img = img / 255.0
img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
img = np.transpose(img, (2, 0, 1))
img = torch.from_numpy(img).unsqueeze(0).float()

model.eval()

features = model.features(img)
features_grad = torch.autograd.grad(outputs=features, inputs=img, grad_outputs=torch.ones_like(features), retain_graph=True, create_graph=True)[0]
weights = torch.mean(features_grad, axis=(2, 3), keepdim=True)

cam = torch.sum(weights * features, axis=1, keepdim=True)
cam = torch.relu(cam)
cam = cam.detach().numpy()[0, 0]
cam = cv2.resize(cam, (img.shape[3], img.shape[2]))
cam = cam - np.min(cam)
cam = cam / np.max(cam)
cam = np.uint8(255 * cam)

heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
result = cv2.addWeighted(cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB), 0.5, heatmap, 0.5, 0)
plt.imshow(result)
plt.show()


