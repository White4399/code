"""
Load pascal VOC dataset
"""

import os

import numpy as np
from PIL import Image
import torch

from .common import BaseDataset

class VOC(BaseDataset):
    """
    Base Class for VOC Dataset

    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._label_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._inst_dir = os.path.join(self._base_dir, 'SegmentationObjectAug')
        self._scribble_dir = os.path.join(self._base_dir, 'ScribbleAugAuto')
        self._id_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        self.transforms = transforms
        self.to_tensor = to_tensor

        with open(os.path.join(self._id_dir, f'{self.split}.txt'), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def remap_mask(self, mask_pil):
        mask = mask_pil.convert('L')
        mask = np.array(mask)

        mapping = {
            0: 0,
            51: 1,
            102: 2,
            153: 3,
            204: 4,
            255: 5
        }

        new_mask = np.zeros_like(mask, dtype=np.uint8)

        for old_val, new_val in mapping.items():
            new_mask[mask == old_val] = new_val

        return Image.fromarray(new_mask)
# 20260417

    def __getitem__(self, idx):
        # # Fetch data
        # id_ = self.ids[idx]
        # image = Image.open(os.path.join(self._image_dir, f'{id_}.jpg'))
        # semantic_mask = Image.open(os.path.join(self._label_dir, f'{id_}.png'))
        # instance_mask = Image.open(os.path.join(self._inst_dir, f'{id_}.png'))
        # # scribble_mask = Image.open(os.path.join(self._scribble_dir, f'{id_}.png'))
        # scribble_mask = Image.fromarray(
        #     np.zeros_like(np.array(semantic_mask), dtype='uint8')
        # )
        # sample = {'image': image,
        #           'label': semantic_mask,
        #           'inst': instance_mask,
        #           'scribble': scribble_mask
        #         }

        # Fetch data
        id_ = self.ids[idx]
        image = Image.open(os.path.join(self._image_dir, f'{id_}.jpg'))
        
        # 加载并转换语义分割标注
        semantic_mask = Image.open(os.path.join(self._label_dir, f'{id_}.png'))

# 20260417
        semantic_mask = self.remap_mask(semantic_mask)
        
        
        # # 转换为单通道灰度图
        # if semantic_mask.mode != 'L':
        #     semantic_mask = semantic_mask.convert('L')
        # # 如果还是4通道，再处理一次
        # semantic_array = np.array(semantic_mask)
        # if len(semantic_array.shape) == 3 and semantic_array.shape[-1] == 4:
        #     semantic_array = semantic_array[:, :, 0]
        #     semantic_mask = Image.fromarray(semantic_array, mode='L')
        
        # # 同样处理实例分割标注
        # instance_mask = Image.open(os.path.join(self._inst_dir, f'{id_}.png'))
        # if instance_mask.mode != 'L':
        #     instance_mask = instance_mask.convert('L')
        # instance_array = np.array(instance_mask)
        # if len(instance_array.shape) == 3 and instance_array.shape[-1] == 4:
        #     instance_array = instance_array[:, :, 0]
        #     instance_mask = Image.fromarray(instance_array, mode='L')
        
        # # 处理scribble标注
        # scribble_path = os.path.join(self._scribble_dir, f'{id_}.png')
        # if os.path.exists(scribble_path):
        #     scribble_mask = Image.open(scribble_path)
        #     if scribble_mask.mode != 'L':
        #         scribble_mask = scribble_mask.convert('L')
        #     scribble_array = np.array(scribble_mask)
        #     if len(scribble_array.shape) == 3 and scribble_array.shape[-1] == 4:
        #         scribble_array = scribble_array[:, :, 0]
        #         scribble_mask = Image.fromarray(scribble_array, mode='L')
        # else:
        #     # 创建全0占位
        #     scribble_mask = Image.fromarray(np.zeros_like(np.array(semantic_mask), dtype='uint8'))
        
         # -------------------------------------------------
        # instance mask
        # -------------------------------------------------
        inst_path = os.path.join(self._inst_dir, f'{id_}.png')
        if os.path.exists(inst_path):
            instance_mask = Image.open(inst_path)
            instance_mask = self.remap_mask(instance_mask)
        else:
            instance_mask = Image.fromarray(
                np.zeros_like(np.array(semantic_mask), dtype=np.uint8)
            )

        # -------------------------------------------------
        # scribble mask
        # -------------------------------------------------
        scribble_path = os.path.join(self._scribble_dir, f'{id_}.png')
        if os.path.exists(scribble_path):
            scribble_mask = Image.open(scribble_path)
            scribble_mask = self.remap_mask(scribble_mask)
        else:
            scribble_mask = Image.fromarray(
                np.zeros_like(np.array(semantic_mask), dtype=np.uint8)
            )



        sample = {'image': image,
                'label': semantic_mask,
                'inst': instance_mask,
                'scribble': scribble_mask}
        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample
