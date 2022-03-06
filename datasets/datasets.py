import random
import torch
import openslide

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.dataset_utils import get_training_augmentation, get_validation_augmentation
from utils.constant import CLASSES_PER_PROVIDER
from utils.dataset_utils import (
    merge_cls,
    parse_csv,
    parse_csv_seg,
    return_random_patch,
    get_segmentation_paths,
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
            self.X, self.y = parse_csv(params.root_dataset, "train")
        else:
            self.X, self.y = parse_csv(params.root_dataset, "test")

    def __len__(self):
        return len(self.X)

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
        img_path = self.X[idx]
        label = self.y[idx]

        wsi_image = openslide.OpenSlide(img_path)
        pil_imgs = [
            return_random_patch(
                wsi_image, self.params.patch_size, self.params.percentage_blank
            )
            for _ in range(self.params.nb_samples)
        ]
        output_tensor = torch.stack([self.transform(pil_img) for pil_img in pil_imgs])

        return output_tensor, label


class BaseSegDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, train=True, transform=None):
        super().__init__()

        self.params = params

        self.path_seg = None

        if train:
            self.X, self.y = parse_csv_seg(
                params.root_dataset, "train", params.data_provider
            )
            self.X, self.path_seg, self.y = get_segmentation_paths(self.X, self.y)
        else:
            self.X, self.y = parse_csv_seg(
                params.root_dataset, "test", params.data_provider
            )

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")


class PatchSegDataset(BaseSegDataset):
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
        img_path = self.X[idx]
        seg_path = self.path_seg[idx]

        wsi_image = openslide.OpenSlide(img_path)
        wsi_seg = openslide.OpenSlide(seg_path)

        pil_imgs = []
        seg_gt = []
        for _ in range(self.params.nb_samples):
            pil_img, seg_img = return_random_patch_with_mask(
                wsi_image, wsi_seg, self.params.patch_size, self.params.percentage_blank
            )

            if self.params.data_provider == "radboud_merged":
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
