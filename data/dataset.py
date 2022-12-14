import torch
import cv2
from typing import List

from albumentations import (
    Compose,
    Normalize,
    Resize,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    RandomRotate90,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    Transpose,
)

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def _get_transforms(use_augmentation: bool, img_size: int):
    if use_augmentation:
        return Compose(
            [
                Transpose(p=0.5),
                Rotate(30, p=0.5),
                RandomRotate90(p=0.5),
                Resize(img_size, img_size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
                ToTensorV2(),
            ]
        )

    return Compose(
        [
            Resize(img_size, img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class LandmarkDataset(Dataset):
    def __init__(
        self,
        img_paths: List,
        labels: List,
        training: bool = True,
        img_size: int = 224,
        use_augmentation: bool = True,
    ):
        self.img_paths = img_paths
        self.labels = labels
        self.training = training
        self.img_size = img_size
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = _get_transforms(self.use_augmentation, self.img_size)(image=img)
        img = augmented["image"]

        if self.training:
            label = self.labels[item]

            return {
                "input": img,
                "target": torch.tensor(label, dtype=torch.long),
            }

        return {
            "input": img,
        }
