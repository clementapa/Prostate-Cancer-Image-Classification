import random
import torch
import openslide
import pandas as pd
import os.path as osp
import os

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.dataset_utils import get_training_augmentation, get_validation_augmentation
from utils.constant import CLASSES_PER_PROVIDER
from utils.dataset_utils import (
    merge_cls,
    return_random_patch,
    return_random_patch_with_mask,
)


class BaseDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, train=True, transform=None):
        super().__init__()

        self.params = params

        if train:
            self.df = pd.read_csv(osp.join(self.params.root_dataset, "train.csv"))
            self.subpath = osp.join("train", "train")
        else:
            self.df = pd.read_csv(osp.join(self.params.root_dataset, "test.csv"))
            self.subpath = osp.join("test", "test")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")


class PatchDataset(BaseDataset):
    def __init__(self, params, train=True, transform=None):
        super().__init__(params, train, transform)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):
        data = dict(self.df.iloc[idx])

        img_path = osp.join(
            self.params.root_dataset, self.subpath, data["image_id"] + ".tiff"
        )

        wsi_image = openslide.OpenSlide(img_path)
        pil_imgs = [
            return_random_patch(
                wsi_image,
                self.params.patch_size,
                self.params.percentage_blank,
                self.params.level,
            )
            for _ in range(self.params.nb_samples)
        ]
        output_tensor = torch.stack([self.transform(pil_img) for pil_img in pil_imgs])

        return output_tensor, data["isup_grade"]


class BaseSegDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, train=True, transform=None):
        super().__init__()

        self.params = params

        if train:
            self.df = pd.read_csv(osp.join(self.params.root_dataset, "train.csv"))
            self.subpath = osp.join("train", "train")
            self.subpath_masks = osp.join("train_label_masks", "train_label_masks")

            name = self.df["image_id"] + ".tiff"
            mask = name.isin(
                os.listdir(osp.join(self.params.root_dataset, self.subpath_masks))
            )
            self.df = self.df[mask].copy()

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")


class PatchSegDataset(BaseSegDataset):
    def __init__(self, params, train=True, transform=None):
        super().__init__(params, train, transform)

        self.params = params
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):
        data = dict(self.df.iloc[idx])

        img_path = osp.join(
            self.params.root_dataset, self.subpath, data["image_id"] + ".tiff"
        )
        wsi_image = openslide.OpenSlide(img_path)

        mask_path = img_path.replace("train", "train_label_masks")
        wsi_seg = openslide.OpenSlide(mask_path)

        pil_imgs = []
        seg_gt = []
        for _ in range(self.params.nb_samples):
            pil_img, seg_img = return_random_patch_with_mask(
                wsi_image,
                wsi_seg,
                self.params.patch_size,
                self.params.percentage_blank,
                self.params.level,
            )
            if data["data_provider"] == "radboud" and self.params.data_provider == "all":
                seg_img = merge_cls(seg_img)
            pil_imgs.append(pil_img)
            seg_gt.append(seg_img)

        output_tensor = torch.stack([self.transform(pil_img) for pil_img in pil_imgs])
        seg_masks = torch.stack(seg_gt)
        return output_tensor, seg_masks


class SegDataset(BaseSegDataset):
    def __init__(self, params, train=True, transform=None):
        super().__init__(params, train, transform)

        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        if train:
            self.transform = get_training_augmentation(
                CLASSES_PER_PROVIDER[params.data_provider]
            )
        else:
            self.transform = get_validation_augmentation(
                CLASSES_PER_PROVIDER[params.data_provider]
            )

    def __getitem__(self, idx):
        img_path = self.X[idx]
        seg_path = self.path_seg[idx]

        wsi_image = openslide.OpenSlide(img_path)
        wsi_seg = openslide.OpenSlide(seg_path)

        resized_img = wsi_image.get_thumbnail(
            (self.params.image_size, self.params.image_size)
        )
        resized_mask = np.array(
            wsi_seg.get_thumbnail((self.params.image_size, self.params.image_size))
        )[:, :, 0].T

        # resized_mask = torch.as_tensor(
        #     np.array(resized_mask.convert("RGB"))[:, :, 0], dtype=torch.int64
        # )

        # resized_mask = transforms.ToTensor()(resized_mask)[:, :, 0]

        if self.transform != None:
            sample = self.transform(image=resized_img, mask=resized_mask)
            resized_img, resized_mask = sample["image"], sample["mask"]

        return resized_img, resized_mask
