# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
# python "C:\Users\USER\Documents\UniversityProjects\PythonProjects\FinalProject\DetectronByFacebook.py"

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries like numpy, json, cv2 etc.
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# load an image
# C:\\Users\\USER\\Pictures\\picc.jpg
im = cv2.imread("C:\\Users\\USER\\Pictures\\road.jpg")
cv2.imshow("Original", im)
cv2.waitKey(0)

# dimensions of the image
height = im.shape[0]
width = im.shape[1]

# Inference with an instance segmentation model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# draw the instance predictions on the image
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Original", out.get_image()[:, :, ::-1])
cv2.waitKey(0)

# Inference with a panoptic segmentation model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

# extract sky segments and ids
sky_segments = []
sky_ids = []
for segment_info in segments_info:
    if segment_info["category_id"] == 40:
        sky_segments.append(segment_info)
        sky_ids.append(segment_info["id"])

# find lowest and highest pixel of the sky
min_sky_y = 0
max_sky_y = 0
for i in range(height):
    for j in range(width):
        if panoptic_seg[i][j] in sky_ids:
            min_sky_y = max(min_sky_y, i)
            max_sky_y = min(max_sky_y, i)

# draw a bounding box around the sky
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), sky_segments)
out2 = v.draw_box((0, max_sky_y, width-1, min_sky_y))
cv2.imshow("Sky", out2.get_image()[:, :, ::-1])
cv2.waitKey(0)

# # extract the sky
# if max_sky_y == 0:
#     print("No sky detected")
#     crop_im = im
# else:
crop_im = im[min_sky_y:height, 0:width]

# Inference with an instance segmentation model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)
outputs = predictor(crop_im)
v = Visualizer(crop_im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Without Sky", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
