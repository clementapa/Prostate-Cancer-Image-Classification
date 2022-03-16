import os
import os.path as osp
import random
import zipfile

import numpy as np
import openslide
import pandas as pd
import torch
import torchvision.transforms as transforms
import wandb
from torch.utils.data import Dataset
from utils.constant import CLASSES_PER_PROVIDER
from utils.dataset_utils import (
    get_training_augmentation,
    get_validation_augmentation,
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
    def __init__(self, params, train=True, transform=None, wb_run=None):
        super().__init__(params, train, transform)

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
            if data["data_provider"] == "radboud":
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


class BaseStaticDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, train=True, transform=None, wb_run=None):
        super().__init__()

        self.params = params

        if train:
            self.name_dataset = osp.basename(self.params.train_artifact).split(":")[0]
            if not os.path.exists(
                os.path.join(self.params.path_patches, self.name_dataset)
            ):
                # check get artifact in agent_utils
                artifact = wandb.run.use_artifact(self.params.train_artifact)
                datadir = artifact.download(root=self.params.path_patches)

                path_to_zip_file = os.path.join(
                    self.params.path_patches, self.name_dataset + ".zip"
                )
                with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(datadir, self.name_dataset))

            self.df = pd.read_csv(osp.join(params.root_dataset, "train" + ".csv"))

        else:
            self.name_dataset = osp.basename(self.params.test_artifact).split(":")[0]
            if not os.path.exists(
                os.path.join(self.params.path_patches, self.name_dataset)
            ):
                # check get artifact in agent_utils
                artifact = wandb.run.use_artifact(self.params.test_artifact)
                datadir = artifact.download(root=self.params.path_patches)

                path_to_zip_file = os.path.join(
                    self.params.path_patches, self.name_dataset + ".zip"
                )
                with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(datadir, self.name_dataset))

            self.df = pd.read_csv(osp.join(params.root_dataset, "test" + ".csv"))
            # raise NotImplementedError(f"To implement!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")


class StaticPatchDataset(BaseStaticDataset):
    def __init__(self, params, train=True, transform=None, wb_run=None):
        super().__init__(params, train, transform, wb_run)
        self.train = train
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

        data = dict(self.df.iloc[idx])

        np_path = osp.join(
            self.params.path_patches, self.name_dataset, data["image_id"] + ".npy"
        )
        np_array = np.load(open(np_path, "rb"))

        if self.train:
            label = data["isup_grade"]
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

class ConcatPatchDataset(BaseStaticDataset):
    def __init__(self, params, train=True, transform=None, wb_run=None):
        super().__init__(params, train, transform, wb_run)
        self.train = train
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((params.resized_patch, params.resized_patch)),
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
                    transforms.Resize((params.resized_patch, params.resized_patch)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
    
    

    def __getitem__(self, idx):

        data = dict(self.df.iloc[idx])

        np_path = osp.join(self.params.path_patches, self.name_dataset, data['image_id'] + ".npy")
        np_array = np.load(open(np_path, "rb"))

        if self.train:
            label = data['isup_grade']
        else:
            label = -1
        images_to_pick = [random.randint(0, np_array.shape[0]-1) for _ in range(self.params.nb_samples)]
        
        output_tensor = torch.stack([self.transform((torch.from_numpy(np_img)/255.0).permute(2,1,0)) for np_img in np_array[images_to_pick]])
        # print(output_tensor.shape)
        
        output_tensor = rearrange(output_tensor, "(n1 n2) c h w -> c (n1 h) (n2 w)", n1=self.params.nb_patches)

        return output_tensor, label