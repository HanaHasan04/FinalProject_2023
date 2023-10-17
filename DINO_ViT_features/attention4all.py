import os
import cv2
import random
import colorsys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from transformers import ViTFeatureExtractor, ViTModel
import torchvision


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


root_dir = r"C:\Users\USER\Documents\UniversityProjects\FinalProject\top_images"
subfolders = [f"S{i}" for i in range(1, 31) if i != 11]

feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vitb8", size=480)
model = ViTModel.from_pretrained("facebook/dino-vitb8", add_pooling_layer=False)

for subfolder in subfolders:
    print(subfolder)
    subfolder_path = os.path.join(root_dir, subfolder)

    emotion_folders = ["Anticipation", "Baseline", "Disappointment", "Frustration"]
    for emotion in emotion_folders:
        print(emotion)
        emotion_folder_path = os.path.join(subfolder_path, emotion, "image")

        if not os.path.exists(emotion_folder_path):
            continue

        image_files = [f for f in os.listdir(emotion_folder_path) if f.endswith(".jpg")]
        for image_file in image_files:
            image_path = os.path.join(emotion_folder_path, image_file)
            image_ = Image.open(image_path).convert('RGB')
            pixel_values = feature_extractor(images=image_, return_tensors="pt").pixel_values

            outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)
            attentions = outputs.attentions[-1]
            nh = attentions.shape[1]
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
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
            th_attn = th_attn.reshape(nh, pixel_values.shape[-2] // model.config.patch_size,
                                      pixel_values.shape[-1] // model.config.patch_size).float()
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu().numpy()

            # TODO
            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = \
            nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[
                0].cpu()
            attentions = attentions.detach().numpy()
            attn = attentions[1]
            attn = torch.tensor(attn)
            height = 480
            width = 480
            grid = attn.view(height, width).detach().numpy()
            heatmap_folder = os.path.join(subfolder_path, emotion, "heatmap")
            fname = os.path.join(heatmap_folder, image_file)
            plt.imsave(fname=fname, arr=grid, format='png', cmap='jet', dpi=224)

            smoothed = cv2.GaussianBlur(grid, (0, 0), sigmaX=30)
            smoothed_normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
            smooth_folder = os.path.join(subfolder_path, emotion, "smooth")
            fname = os.path.join(smooth_folder, image_file)
            plt.imsave(fname=fname, arr=smoothed_normalized, format='png', cmap='jet', dpi=224)

            # overlaid_image = Image.blend(image_, smoothed_normalized, alpha=0.4)
            # overlaid_folder = os.path.join(subfolder_path, emotion, "overlaid")
            # os.makedirs(overlaid_folder, exist_ok=True)
            # overlaid_image.save(os.path.join(overlaid_folder, image_file))

            base_image = Image.open(image_path)
            smooth_image = Image.open(fname)
            if base_image.mode != 'RGB':
                base_image = base_image.convert('RGB')
            if smooth_image.mode != 'RGB':
                smooth_image = smooth_image.convert('RGB')
            # Ensure both images have the same dimensions (resize if necessary)
            if base_image.size != smooth_image.size:
                smooth_image = smooth_image.resize(base_image.size)
            overlaid_image = Image.blend(base_image, smooth_image, alpha=0.4)
            overlaid_folder = os.path.join(subfolder_path, emotion, "overlaid")
            os.makedirs(overlaid_folder, exist_ok=True)
            overlaid_image.save(os.path.join(overlaid_folder, image_file))
            # TODO


            # heatmap_folder = os.path.join(subfolder_path, emotion, "heatmap")
            # os.makedirs(heatmap_folder, exist_ok=True)
            # heatmap_image = Image.fromarray((th_attn[0] * 255).astype(np.uint8))
            # heatmap_image.save(os.path.join(heatmap_folder, image_file))

            # im = th_attn[1]
            # im = torch.tensor(im)
            # height, width = 480, 480
            # grid = im.view(height, width).detach().numpy()
            # smoothed = cv2.GaussianBlur(grid, (0, 0), sigmaX=30)
            # smoothed_normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

            # smooth_folder = os.path.join(subfolder_path, emotion, "smooth")
            # os.makedirs(smooth_folder, exist_ok=True)
            # smooth_image = Image.fromarray((smoothed_normalized * 255).astype(np.uint8))
            # smooth_image.save(os.path.join(smooth_folder, image_file))

            # base_image = Image.open(image_path)
            # if base_image.mode != 'RGB':
            #     base_image = base_image.convert('RGB')
            # if smooth_image.mode != 'RGB':
            #     smooth_image = smooth_image.convert('RGB')
            # # Ensure both images have the same dimensions (resize if necessary)
            # if base_image.size != smooth_image.size:
            #     smooth_image = smooth_image.resize(base_image.size)
            # overlaid_image = Image.blend(base_image, smooth_image, alpha=0.4)
            # overlaid_folder = os.path.join(subfolder_path, emotion, "overlaid")
            # os.makedirs(overlaid_folder, exist_ok=True)
            # overlaid_image.save(os.path.join(overlaid_folder, image_file))

print("Done for all horses.")
