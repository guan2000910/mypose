from typing import Union

import cv2
import numpy as np
import torch
import albumentations as A

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps

import random

#图像处理与增强
imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)#定义的均值和标准差

#归一化
def normalize(img: np.ndarray):  # (h, w, 3) -> (3, h, w)
    mu, std = imagenet_stats
    if img.dtype == np.uint8:
        img = img / 255
    img = (img - mu) / std
    return img.transpose(2, 0, 1).astype(np.float32) 


def denormalize(img: Union[np.ndarray, torch.Tensor]): #反归一化
    mu, std = imagenet_stats
    if isinstance(img, torch.Tensor):
        mu, std = [torch.Tensor(v).type(img.dtype).to(img.device)[:, None, None] for v in (mu, std)]
    return img * std + mu


class Unsharpen(A.ImageOnlyTransform): #图像增强 去锐化 随机选择锐化核的大小和强度，对图像进行高斯模糊并与原图像进行加权融合，得到去锐化后的图像。
    def __init__(self, k_limits=(3, 7), strength_limits=(0., 2.), p=0.5):
        super().__init__()
        self.k_limits = k_limits #锐化核大小从3到7
        self.strength_limits = strength_limits #锐化强度
        self.p = p #去锐化处理概率

    def apply(self, img, **params): 
        if np.random.rand() > self.p: 
            return img           #不执行锐化
        k = np.random.randint(self.k_limits[0] // 2, self.k_limits[1] // 2 + 1) * 2 + 1
        s = k / 3
        blur = cv2.GaussianBlur(img, (k, k), s)
        strength = np.random.uniform(*self.strength_limits)
        unsharpened = cv2.addWeighted(img, 1 + strength, blur, -strength, 0)
        return unsharpened


class DebayerArtefacts(A.ImageOnlyTransform): #图像增强
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def apply(self, img, **params):
        if np.random.rand() > self.p:
            return img
        assert img.dtype == np.uint8
        # permute channels before bayering/debayering to cover different bayer formats
        channel_idxs = np.random.permutation(3)
        channel_idxs_inv = np.empty(3, dtype=int)
        channel_idxs_inv[channel_idxs] = 0, 1, 2

        # assemble bayer image
        bayer = np.zeros(img.shape[:2], dtype=img.dtype)
        bayer[::2, ::2] = img[::2, ::2, channel_idxs[2]]
        bayer[1::2, ::2] = img[1::2, ::2, channel_idxs[1]]
        bayer[::2, 1::2] = img[::2, 1::2, channel_idxs[1]]
        bayer[1::2, 1::2] = img[1::2, 1::2, channel_idxs[0]]

        # debayer
        debayer_method = np.random.choice((cv2.COLOR_BAYER_BG2BGR, cv2.COLOR_BAYER_BG2BGR_EA))
        debayered = cv2.cvtColor(bayer, debayer_method)[..., channel_idxs_inv]
        return debayered

class ColorEnhancement(A.ImageOnlyTransform):
    def __init__(self, brightness_limits=(0.1, 0.3), contrast_limits=(-0.2, 0.2), saturation_limits=(-0.2, 0.2), hue_limits=(-0.05, 0.05), p=0.5):
        super().__init__()
        self.brightness_limits = brightness_limits
        self.contrast_limits = contrast_limits
        self.saturation_limits = saturation_limits
        self.hue_limits = hue_limits
        self.p = p

    def apply(self, img, **params):
        if np.random.rand() > self.p:
            return img

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        brightness = np.random.uniform(*self.brightness_limits)
        contrast = np.random.uniform(*self.contrast_limits)
        saturation = np.random.uniform(*self.saturation_limits)
        hue = np.random.uniform(*self.hue_limits)

        img_hsv[..., 0] = np.clip(img_hsv[..., 0] + int(np.round(179 * hue)), 0, 179).astype(np.uint8)
        img_hsv[..., 0] = np.clip(img_hsv[..., 0], 0, 179).astype(np.uint8)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + 255 * saturation, 0, 255).astype(np.uint8)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * (1 + contrast), 0, 255).astype(np.uint8)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + 255 * brightness, 0, 255).astype(np.uint8)

        img_enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_enhanced

