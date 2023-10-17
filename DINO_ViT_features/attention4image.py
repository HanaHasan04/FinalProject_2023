import os
import cv2
import random
import colorsys
import numpy as np
import torch
import torch.nn as nn
import requests
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from transformers import ViTFeatureExtractor, ViTModel
import torchvision
import torch.nn.functional as F


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    colors = random_colors(N)

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))

        masked_image = apply_mask(masked_image, _mask, color, alpha)

        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)

            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


# url_png = 'grayst_Frustration_framestep01.png'
# url = 'grayst_Frustration_framestep01.jpg'
url_png = r"C:\Users\USER\Documents\UniversityProjects\FinalProject\grayst_Frustration_framestep01.png"
url = r"C:\Users\USER\Documents\UniversityProjects\FinalProject\grayst_Frustration_framestep01.jpg"
image_ = Image.open(url).convert('RGB')
image = image_.save(url_png)
image = Image.open(url_png).convert('RGB')


feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vitb8", size=480)

pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
print(pixel_values.shape)


model = ViTModel.from_pretrained("facebook/dino-vitb8", add_pooling_layer=False)
outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)

# Visualize
attentions = outputs.attentions[-1]
nh = attentions.shape[1]
nh

attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
print(attentions.shape)

threshold = 0.6
w_featmap = pixel_values.shape[-2] // model.config.patch_size
h_featmap = pixel_values.shape[-1] // model.config.patch_size

val, idx = torch.sort(attentions)
val /= torch.sum(val, dim=1, keepdim=True)
cumval = torch.cumsum(val, dim=1)
th_attn = cumval > (1 - threshold)
idx2 = torch.argsort(idx)
for head in range(nh):
    th_attn[head] = th_attn[head][idx2[head]]
th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[
    0].cpu().numpy()

attentions = attentions.reshape(nh, w_featmap, h_featmap)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[
    0].cpu()
attentions = attentions.detach().numpy()

output_dir = '.'
os.makedirs(output_dir, exist_ok=True)
torchvision.utils.save_image(torchvision.utils.make_grid(pixel_values, normalize=True, scale_each=True),
                             os.path.join(output_dir, "img.png"))
for j in range(nh):
    fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
    plt.figure()
    plt.imshow(attentions[j])
    plt.imsave(fname=fname, arr=attentions[j], format='png')

im = attentions[1]
im = torch.tensor(im)
height = 480
width = 480

grid = im.view(height, width).detach().numpy()

smoothed = cv2.GaussianBlur(grid, (0, 0), sigmaX=30)

smoothed_normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(grid, cmap='jet')

fname = os.path.join(output_dir, "attn.png")
plt.imsave(fname=fname, arr=grid, format='png', cmap='jet')

plt.subplot(1, 2, 2)
plt.title('Smoothed')
plt.imshow(smoothed_normalized, cmap='jet')

plt.show()

fname = os.path.join(output_dir, "map.png")
plt.imsave(fname=fname, arr=smoothed_normalized, format='png', cmap='jet')

base = Image.open('grayst_Frustration_framestep01.png')
mask = Image.open('map.png')

if base.mode != 'RGB':
    base = base.convert('RGB')

if mask.mode != 'RGB':
    mask = mask.convert('RGB')

base = base.resize((480, 480), Image.LANCZOS)
mask = mask.resize((480, 480), Image.LANCZOS)

result = Image.blend(base, mask, alpha=0.4)
result.show()

# Save all

base = Image.open('grayst_Frustration_framestep01.png')
base = base.resize((224, 224), Image.LANCZOS)
base.save('base.png')

attn = Image.open('attn.png')
attn = attn.resize((224, 224), Image.LANCZOS)
attn.save('attn.png')

smooth = Image.open('map.png')
smooth = smooth.resize((224, 224), Image.LANCZOS)
smooth.save('smooth.png')

if base.mode != 'RGB':
    base = base.convert('RGB')

if mask.mode != 'RGB':
    mask = mask.convert('RGB')

if smooth.mode != 'RGB':
    smooth = smooth.convert('RGB')

blend = Image.blend(base, smooth, alpha=0.4)
blend = blend.resize((224, 224), Image.LANCZOS)
blend.save('blend.png')
