"""
Ultrasound dataset for few-shot segmentation.
Builds the same batch structure as VOC (via customized.fewShot).
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _palette_pixel_to_class_lut():
    """PNG 像素 0,51,...,255 -> 语义类别 0..5；其余像素值映射为背景 0。"""
    lut = np.zeros(256, dtype=np.uint8)
    for class_id, pix in enumerate((0, 51, 102, 153, 204, 255)):
        lut[int(pix)] = class_id
    return lut


# 供 transforms 等模块 `from .m_ultrasound import ...` 使用（与 UltrasoundToTensorNormalize 一致）
PALETTE_6IDX_PIXEL_TO_CLASS = _palette_pixel_to_class_lut()


def label_array_to_semantic_raw(arr: np.ndarray) -> np.ndarray:
    """
    标注 numpy 数组 -> (H,W) uint8，取值 0..5。
    若为 RGBA/RGB，使用第 0 通道（与现有处理一致）。
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    a = np.clip(a.astype(np.int64), 0, 255)
    return PALETTE_6IDX_PIXEL_TO_CLASS[a].astype(np.uint8)


class UltrasoundFewShotDataset(Dataset):
    """
    目录（与 VOC 布局兼容，便于共用 JPEGImages / ImageSets）:
        base_dir/
        ├── JPEGImages/
        ├── SegmentationClassAug/   # 语义标注（像素 0,51,...,255）
        ├── ScribbleAugAuto/        # 涂鸦（可为全 0；当前 few-shot 用 dense mask，不依赖实例图）
        └── ImageSets/Segmentation/
            ├── train.txt / val.txt  # 每行一个 id，无扩展名
    """

    def __init__(
        self,
        base_dir,
        split="train",
        transforms=None,
        to_tensor=None,
        labels=None,
        max_iters=None,
        n_ways=5,
        n_shots=1,
        n_queries=1,
        input_size=(480, 640),
        mask_dir_name="SegmentationClassAug",
        scribble_dir_name="ScribbleAugAuto",
        use_red_channel=False,
    ):
        self.base_dir = base_dir
        self.split = split
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.input_size = input_size
        self.labels_filter = labels if labels is not None else [1, 2, 3, 4, 5]
        self.use_red_channel = use_red_channel

        self.class_mapping = {0: 0, 51: 1, 102: 2, 153: 3, 204: 4, 255: 5}
        self.num_classes = 6  # 与 PALETTE_6IDX_PIXEL_TO_CLASS 一致

        self.img_dir = os.path.join(base_dir, "JPEGImages")
        self.mask_dir = os.path.join(base_dir, mask_dir_name)
        self.scribble_dir = os.path.join(base_dir, scribble_dir_name)

        split_file = os.path.join(base_dir, "ImageSets", "Segmentation", f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                self.img_names = [line.strip() for line in f if line.strip()]
        else:
            self.img_names = [
                os.path.splitext(f)[0]
                for f in os.listdir(self.img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        self.class_images = {i: [] for i in range(1, self.num_classes)}
        for img_name in self.img_names:
            mask_path = os.path.join(self.mask_dir, f"{img_name}.png")
            if not os.path.exists(mask_path):
                continue
            mask_arr = label_array_to_semantic_raw(np.array(Image.open(mask_path)))
            for cls_id in np.unique(mask_arr):
                cls_id = int(cls_id)
                if cls_id > 0 and cls_id in self.labels_filter:
                    if img_name not in self.class_images[cls_id]:
                        self.class_images[cls_id].append(img_name)

        self.valid_classes = [
            c for c in range(1, self.num_classes) if len(self.class_images[c]) > 0
        ]
        if len(self.valid_classes) < n_ways:
            raise RuntimeError(
                f"Need at least n_ways={n_ways} classes with pixels in split '{split}', "
                f"got valid_classes={self.valid_classes}"
            )

        if max_iters is not None:
            self.max_iters = max_iters
        else:
            self.max_iters = max(
                1, len(self.img_names) // max(1, n_ways * (n_shots + n_queries)) * 10
            )

        print(f"Ultrasound [{split}]: {len(self.img_names)} ids, valid_classes={self.valid_classes}")

    def __len__(self):
        return self.max_iters

    def _open_image(self, img_name):
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(self.img_dir, f"{img_name}{ext}")
            if os.path.exists(p):
                img = Image.open(p)
                if self.use_red_channel:
                    arr = np.array(img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB"))
                    if arr.ndim == 3 and arr.shape[2] >= 1:
                        r = arr[:, :, 0]
                        img = Image.fromarray(np.stack([r, r, r], axis=-1).astype(np.uint8))
                else:
                    img = img.convert("RGB")
                return img
        raise FileNotFoundError(f"No image for id {img_name} under {self.img_dir}")

    def _open_mask_pil(self, path):
        if not os.path.exists(path):
            return None
        return Image.open(path)

    def _build_sample_tensor(self, img_name, basic_class_id):
        """与 VOC __getitem__ 末尾一致：PIL -> transforms -> to_tensor -> basic_class_id"""
        image = self._open_image(img_name)
        label = self._open_mask_pil(os.path.join(self.mask_dir, f"{img_name}.png"))
        if label is None:
            raise FileNotFoundError(os.path.join(self.mask_dir, f"{img_name}.png"))

        # 实例图常为实例 ID，不经语义调色板映射；非 bbox 模式用全零即可
        inst = Image.new("L", image.size, 0)
        scribble_path = os.path.join(self.scribble_dir, f"{img_name}.png")
        scribble = self._open_mask_pil(scribble_path)
        if scribble is None:
            scribble = Image.new("L", image.size, 0)

        sample = {
            "image": image,
            "label": label,
            "inst": inst,
            "scribble": scribble,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        image_t = torch.from_numpy(np.array(sample["image"]).transpose(2, 0, 1))
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)
        sample["image_t"] = image_t
        sample["id"] = img_name#smh0521
        sample["basic_class_id"] = basic_class_id
        return sample

    # def __getitem__(self, index):
    #     from .customized import fewShot

    #     selected_classes = random.sample(self.valid_classes, self.n_ways)
    #     # print(f"[Episode {index}] selected_classes={selected_classes}")  # ← 加这行验证
    #     cnt_query = np.bincount(
    #         random.choices(population=range(self.n_ways), k=self.n_queries),
    #         minlength=self.n_ways,
    #     )
    #     paired_sample = []
    #     for way, cls_id in enumerate(selected_classes):
    #         need = self.n_shots + int(cnt_query[way])
    #         pool = self.class_images[cls_id]
    #         if len(pool) < need:
    #             chosen = random.choices(pool, k=need)
    #         else:
    #             chosen = random.sample(pool, k=need)
    #         # print(f"[Episode {index}] way={way} cls={cls_id} chosen={chosen}")  # 打印采样到的图片名
    #         for _img in chosen:
    #             paired_sample.append(self._build_sample_tensor(_img, cls_id))

    #     return fewShot(
    #         paired_sample,
    #         n_ways=self.n_ways,
    #         n_shots=self.n_shots,
    #         cnt_query=cnt_query,
    #         coco=False,
    #     )
    def __getitem__(self, index):
        from .customized import fewShot

        rng = random.Random(hash((index, id(self))))

        selected_classes = rng.sample(self.valid_classes, self.n_ways)
        cnt_query = np.bincount(
            rng.choices(population=range(self.n_ways), k=self.n_queries),
            minlength=self.n_ways,
        )
        paired_sample = []
        for way, cls_id in enumerate(selected_classes):
            need = self.n_shots + int(cnt_query[way])
            pool = self.class_images[cls_id]
            if len(pool) < need:
                chosen = rng.choices(pool, k=need)
            else:
                chosen = rng.sample(pool, k=need)
            for _img in chosen:
                paired_sample.append(self._build_sample_tensor(_img, cls_id))

        return fewShot(
            paired_sample,
            n_ways=self.n_ways,
            n_shots=self.n_shots,
            cnt_query=cnt_query,
            coco=False,
        )

def ultrasound_fewshot(
    base_dir,
    split,
    transforms,
    to_tensor,
    labels,
    max_iters,
    n_ways,
    n_shots,
    n_queries,
    input_size,
):
    return UltrasoundFewShotDataset(
        base_dir=base_dir,
        split=split,
        transforms=transforms,
        to_tensor=to_tensor,
        labels=labels,
        max_iters=max_iters,
        n_ways=n_ways,
        n_shots=n_shots,
        n_queries=n_queries,
        input_size=input_size,
    )


