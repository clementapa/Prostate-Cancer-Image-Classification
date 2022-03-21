import os.path as osp
import random

from math import sqrt

import numpy as np
import openslide
import torch
import torchvision.transforms as transforms
from einops import rearrange
from tqdm import tqdm

from utils.dataset_utils import (
    get_training_augmentation,
    get_validation_augmentation,
    merge_cls,
    return_random_patch,
    return_random_patch_with_mask,
    seg_max_to_score,
)

import utils.agent_utils as au
from datasets.BaseDatasets import BaseDataset


class PatchDataset(BaseDataset):
    def __init__(self, params, X, y, df, train=True):
        super().__init__(params, X, y, df, train)

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, idx):
        image_id = self.X[idx]

        img_path = osp.join(self.params.root_dataset, self.subpath, image_id + ".tiff")

        if self.train:
            label = self.y[idx]
        else:
            label = -1

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

        return output_tensor, label


class StaticPatchDataset(BaseDataset):
    def __init__(self, params, X, y, df, train=True):
        super().__init__(params, X, y, df, train)

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, idx):

        image_id = self.X[idx]

        np_path = osp.join(self.params.patch_folder, image_id + ".npy")
        np_array = np.load(open(np_path, "rb"))

        if self.train:
            label = self.y[idx]
        else:
            label = -1

        # images_to_pick = [random.randint(0, np_array.shape[0]-1) for _ in range(self.params.nb_samples)] # tirage avec remise

        images_to_pick = random.sample(
            [i for i in range(self.params.nb_samples)], self.params.nb_samples
        )  # tirage sans remise

        output_tensor = torch.stack(
            [
                self.transform((torch.from_numpy(np_img) / 255.0).permute(2, 1, 0))
                for np_img in np_array[images_to_pick]
            ]
        )

        return output_tensor, label


class ConcatTopPatchDataset(BaseDataset):
    def __init__(self, params, X, y, df, train=True):
        super().__init__(params, X, y, df, train)

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # load seg_model
        self.seg_model = au.get_seg_model(params)

        self.select_top_patches()

    def __getitem__(self, idx):

        image_id = self.X[idx]

        np_path = osp.join(self.params.patch_folder, image_id + ".npy")
        np_array = np.load(open(np_path, "rb"))

        if self.train:
            label = self.y[idx]
        else:
            label = -1

        output_tensor = torch.stack(
            [
                self.transform((torch.from_numpy(np_img) / 255.0).permute(2, 1, 0))
                for np_img in np_array[self.images_to_pick[image_id]]
            ]
        )
        output_tensor = rearrange(
            output_tensor,
            "(n1 n2) c h w -> c (n1 h) (n2 w)",
            n1=int(sqrt(self.params.nb_samples)),
        )

        return output_tensor, label

    @torch.no_grad()
    def select_top_patches(self):

        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.images_to_pick = {}

        for image_id in tqdm(self.X):
            np_path = osp.join(self.params.patch_folder, image_id + ".npy")
            np_array = np.load(open(np_path, "rb"))
            output_tensor = torch.stack(
                [
                    transform((torch.from_numpy(np_img) / 255.0).permute(2, 1, 0))
                    for np_img in np_array
                ]
            )
            seg_masks = self.seg_model(output_tensor.to(device)).argmax(dim=1)
            seg_scores = seg_max_to_score(seg_masks, self.params.patch_size)

            top_k = torch.topk(
                seg_scores[:, -1], min(self.params.nb_samples, output_tensor.shape[0])
            ).indices

            self.images_to_pick[image_id] = top_k.cpu().numpy().tolist()


class ConcatPatchDataset(BaseDataset):
    def __init__(self, params, X, y, df, train=True):
        super().__init__(params, X, y, df, train)

        if train:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((params.resized_patch, params.resized_patch)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((params.resized_patch, params.resized_patch)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, idx):

        image_id = self.X[idx]

        np_path = osp.join(self.params.patch_folder, image_id + ".npy")
        np_array = np.load(open(np_path, "rb"))

        if self.train:
            label = self.y[idx]
        else:
            label = -1

        if self.params.discounted_draw:
            images_to_pick = [
                random.randint(0, np_array.shape[0] - 1)
                for _ in range(self.params.nb_samples)
            ]  # tirage avec remise
        else:
            images_to_pick = random.sample(
                [i for i in range(min(self.params.nb_samples, np_array.shape[0] - 1))],
                min(self.params.nb_samples, np_array.shape[0] - 1),
            )  # tirage sans remise

        output_tensor = torch.stack(
            [
                self.transform((torch.from_numpy(np_img) / 255.0).permute(2, 1, 0))
                for np_img in np_array[images_to_pick]
            ]
        )

        if not self.params.discounted_draw:
            if len(images_to_pick) != self.params.nb_samples:
                diff = self.params.nb_samples - len(images_to_pick)
                diff_tensor = torch.ones_like(
                    torch.randn(
                        diff,
                        output_tensor.shape[1],
                        output_tensor.shape[2],
                        output_tensor.shape[3],
                    )
                )
                output_tensor = torch.cat([output_tensor, diff_tensor], axis=0)
                output_tensor = output_tensor[
                    torch.randperm(self.params.nb_samples)
                ]  # shuffle

        output_tensor = rearrange(
            output_tensor,
            "(n1 n2) c h w -> c (n1 h) (n2 w)",
            n1=int(sqrt(self.params.nb_samples)),
        )

        return output_tensor, label


class PatchSegDataset(BaseDataset):
    def __init__(self, params, X, y, df, train=True):
        super().__init__(params, X, y, df, train)

        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):

        image_id = self.X[idx]

        img_path = osp.join(self.params.root_dataset, self.subpath, image_id + ".tiff")
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
            if (
                self.df[self.df["image_id"] == image_id]["data_provider"].iloc[0]
                == "radboud"
            ):
                seg_img = merge_cls(seg_img)
            pil_imgs.append(pil_img)
            seg_gt.append(seg_img)

        output_tensor = torch.stack([self.transform(pil_img) for pil_img in pil_imgs])
        seg_masks = torch.stack(seg_gt)
        return output_tensor, seg_masks


# class SegDataset(BaseDataset):
#     def __init__(self, params, train=True, transform=None):
#         super().__init__(params, train, transform)

#         if train:
#             self.transform = get_training_augmentation(
#                 CLASSES_PER_PROVIDER[params.data_provider]
#             )
#         else:
#             self.transform = get_validation_augmentation(
#                 CLASSES_PER_PROVIDER[params.data_provider]
#             )

#     def __getitem__(self, idx):
#         image_id = self.X[idx]

#         img_path = osp.join(
#             self.params.root_dataset, self.subpath, image_id + ".tiff"
#         )
#         wsi_image = openslide.OpenSlide(img_path)

#         mask_path = img_path.replace("train", "train_label_masks")
#         wsi_seg = openslide.OpenSlide(mask_path)

#         resized_img = wsi_image.get_thumbnail(
#             (self.params.image_size, self.params.image_size)
#         )
#         resized_mask = np.array(
#             wsi_seg.get_thumbnail((self.params.image_size, self.params.image_size))
#         )[:, :, 0].T

#         # resized_mask = torch.as_tensor(
#         #     np.array(resized_mask.convert("RGB"))[:, :, 0], dtype=torch.int64
#         # )

#         # resized_mask = transforms.ToTensor()(resized_mask)[:, :, 0]

#         if self.transform != None:
#             sample = self.transform(image=resized_img, mask=resized_mask)
#             resized_img, resized_mask = sample["image"], sample["mask"]

#         return resized_img, resized_mask
