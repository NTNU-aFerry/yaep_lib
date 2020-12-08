#!/usr/bin/env python
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

import cv2
import numpy as np

from .backbone import EfficientDetBackbone
from .efficientdet.utils import BBoxTransform, ClipBoxes
from .utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box, aspectaware_resize_padding

force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = []

def set_cuda_device(device_num):
    """
    Sets which CUDA device to be used.

    :param device_num: the CUDA device number
    :type device_num: int
    """
    torch.cuda.set_device(device_num)

def load_model(weights_path, compound_coef):
    """
    Loads the pre-trained weights from the weights folder.

    :param weights_path: path to the weights directory
    :type weights_path: str
    :param compound_coef: coefficient for deceiding pre-trained model
    :type compound_coef: int

    :return: EfficientDetBackbone: EfficientDet model.
    """
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)

    model.load_state_dict(torch.load(weights_path + f'efficientdet-d{compound_coef}.pth', map_location='cpu'))

    model.to('cuda')
    model.cuda()
    model.requires_grad_(False)
    model.eval()

    return model


def preprocess_image(images, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Processes the images into format that the YAEP model excpects.

    :type images: list
    :param images: list of images to pre-process
    :type max_size: int
    :param max_size: the max size of the dimensions of the image.
    :type mean: tuple
    :param mean: three dimensional tuple containing mean RGB values
    from the dataset the weights are trained on.
    :type std: tuple
    :param std: three dimensional tuple containing the standard deviation for RGB values

    :rtype: tuple
    :return: (ori_imgs, framed_imgs, framed_metas)

    """

    ori_imgs = images
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]

    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size, means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def infer_images(images, model, detection_threshold):
    """
    Perform inference on images.

    :type images: list
    :param images: list of images to perform inference on
    :type model: EfficientDetBackbone
    :param model: EfficientDet model that performs the inference
    :type detection_threshold: float
    :param detection_threshold: Threshold of filtering out detections

    :rtype: tuple
    :return: (out, ori_imgs)
    """
    ori_imgs, framed_imgs, framed_metas = preprocess_image(images)

    threshold = detection_threshold
    iou_threshold = detection_threshold

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x, anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    return out, ori_imgs


def display(preds, imgs):
    """
    Displays the image with detections

    :type preds: list
    :param preds: list of predictions
    :type imgs: list
    :param imgs: list of images that the predictions are from

    """
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            if obj == 'person' or obj == 'boat':
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                             color=color_list[get_index_label(obj, obj_list)])

        cv2.imshow('img', imgs[i])
        cv2.waitKey(1)
