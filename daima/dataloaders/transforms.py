"""
Customized data transforms
"""
import random

from PIL import Image, ImageFilter
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F

from .m_ultrasound import PALETTE_6IDX_PIXEL_TO_CLASS, label_array_to_semantic_raw


class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
            scribble = scribble.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)
        inst = tr_F.resize(inst, self.size, interpolation=Image.NEAREST)
        scribble = tr_F.resize(scribble, self.size, interpolation=Image.LANCZOS)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

class DilateScribble(object):
    """
    Dilate the scribble mask

    Args:
        size: window width
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        scribble = sample['scribble']
        dilated_scribble = Image.fromarray(
            ndimage.minimum_filter(np.array(scribble), size=self.size))
        dilated_scribble.putpalette(scribble.getpalette())

        sample['scribble'] = dilated_scribble
        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()
        inst = torch.Tensor(np.array(inst)).long()
        scribble = torch.Tensor(np.array(scribble)).long()

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


# 20260508
class UltrasoundToTensorNormalize(object):
    """
    Convert ultrasound images/masks to torch.Tensor
    专门处理 RGBA 格式的标注，将像素值映射到 0-5（与 m_ultrasound.PALETTE_6IDX_PIXEL_TO_CLASS 一致）
    """
    def __init__(self):
        self.pixel_to_class_lut = PALETTE_6IDX_PIXEL_TO_CLASS

    def _label_to_class_tensor(self, x):
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.array(x)
        mapped = label_array_to_semantic_raw(arr)
        return torch.from_numpy(mapped).long()

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']

        # 处理原图
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 处理标注（RGBA 格式）
        if isinstance(label, dict):
            label = {catId: self._label_to_class_tensor(x) for catId, x in label.items()}
        else:
            label = self._label_to_class_tensor(label)

        # 处理 instance mask（如果有）
        inst_arr = np.array(inst)
        if len(inst_arr.shape) == 3 and inst_arr.shape[2] == 4:
            inst_arr = inst_arr[:, :, 0]
        inst = torch.from_numpy(inst_arr).long()

        # 处理 scribble
        scribble_arr = np.array(scribble)
        if len(scribble_arr.shape) == 3 and scribble_arr.shape[2] == 4:
            scribble_arr = scribble_arr[:, :, 0]
        scribble = torch.from_numpy(scribble_arr).long()

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


class ColorJitter(object):
    """
    随机调整图像亮度和对比度
    适合超声图像，因为超声图像对比度通常较低
    """
    def __init__(self, brightness=0.15, contrast=0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            factor_b = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = tr_F.adjust_brightness(img, factor_b)
        if random.random() < 0.5:
            factor_c = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = tr_F.adjust_contrast(img, factor_c)
        sample['image'] = img
        return sample


class RandomRotation(object):
    """
    随机小幅旋转图像和标注
    适合超声图像，因为探头方向会有变化
    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']

        if random.random() < 0.5:
            angle = random.uniform(-self.max_angle, self.max_angle)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            if isinstance(label, dict):
                label = {catId: x.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                         for catId, x in label.items()}
            else:
                label = label.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            inst = inst.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            scribble = scribble.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


class RandomScale(object):
    """
    随机缩放图像和标注
    适合超声图像，因为探头深度会有变化
    """
    def __init__(self, min_scale=0.85, max_scale=1.15):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']

        if random.random() < 0.5:
            scale = random.uniform(self.min_scale, self.max_scale)
            w, h = img.size
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            img = img.resize((w, h), resample=Image.BILINEAR)
            if isinstance(label, dict):
                label = {catId: (x.resize((new_w, new_h), resample=Image.NEAREST)
                                .resize((w, h), resample=Image.NEAREST))
                         for catId, x in label.items()}
            else:
                label = label.resize((new_w, new_h), resample=Image.NEAREST).resize((w, h), resample=Image.NEAREST)
            inst = inst.resize((new_w, new_h), resample=Image.NEAREST).resize((w, h), resample=Image.NEAREST)
            scribble = scribble.resize((new_w, new_h), resample=Image.NEAREST).resize((w, h), resample=Image.NEAREST)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


class GaussianNoise(object):
    """
    添加高斯噪声，模拟超声图像的散斑噪声
    """
    def __init__(self, mean=0.0, std=0.015, prob=0.25):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, sample):
        img = sample['image']
        if random.random() < self.prob:
            arr = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(self.mean, self.std, arr.shape).astype(np.float32)
            noisy = np.clip(arr + noise, 0.0, 1.0)
            img = Image.fromarray((noisy * 255).astype(np.uint8))
        sample['image'] = img
        return sample


class RandomGaussianBlur(object):
    """
    轻微高斯模糊，增强模型对不同图像质量的鲁棒性
    """
    def __init__(self, prob=0.2, radius_range=(0.5, 1.2)):
        self.prob = prob
        self.radius_range = radius_range

    def __call__(self, sample):
        img = sample['image']
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        sample['image'] = img
        return sample


class UltrasoundResize(object):
    """
    专门为超声数据调整尺寸，保持标注的类别值不变
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        
        img = tr_F.resize(img, self.size)
        
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)
        
        inst = tr_F.resize(inst, self.size, interpolation=Image.NEAREST)
        scribble = tr_F.resize(scribble, self.size, interpolation=Image.NEAREST)
        
        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


