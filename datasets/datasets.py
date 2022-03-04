import random
import torch

import torch.nn.functional as F
import torchvision.transforms as transforms

from einops import rearrange
from tifffile import imread
from torch.utils.data import Dataset

from utils.dataset_utils import parse_csv


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

        # self.transform = transforms.Compose(
        #     [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        # )

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")


class PatchDataset(BaseDataset):
    def __init__(self, params, train=True, transform=None):
        super().__init__(params, train, transform)

    def __getitem__(self, idx):
        img_path = self.X[idx]
        label = self.y[idx]

        img = imread(img_path, key=self.params.key)
        img = torch.from_numpy(img)

        # Square image 
        quantity_to_pad = abs(img.shape[0] - img.shape[1])
        bool_temp = img.shape[1] < img.shape[0]
        img = F.pad(
            img,
            pad=(
                0,
                0,
                quantity_to_pad * bool_temp,
                0,
                quantity_to_pad * (1 - bool_temp),
                0,
            ),
            mode="constant",
            value=255,
        ).unsqueeze(0)

        assert (
            img.shape[1] == img.shape[2]
        )  # check that it is a square image

        # process image to divide per patch
        remaining_pixels = img.shape[1] % self.params.patch_size
        if remaining_pixels != 0:
            if (
                img.shape[1] + remaining_pixels
            ) % self.params.patch_size == 0:
                # padd
                img = F.pad(
                    img,
                    pad=(
                        0,
                        0,
                        remaining_pixels // 2,
                        remaining_pixels // 2,
                        remaining_pixels // 2,
                        remaining_pixels // 2,
                    ),
                    mode="constant",
                    value=255,
                )
            else:
                # crop
                img = img[
                    :,
                    0 : img.shape[1] - remaining_pixels,
                    0 : img.shape[2] - remaining_pixels,
                    :,
                ]

        # Divide image per patch
        h = img.shape[1] // self.params.patch_size
        w = img.shape[2] // self.params.patch_size
        output_patches = rearrange(
            img,
            "b (h p1) (w p2) c -> b (h w) p1 p2 c",
            p1=self.params.patch_size,
            p2=self.params.patch_size,
            h=h,
            w=w,
        )

        # Remove white patches
        mask = (1.0 * (output_patches == 255)).sum(dim=(2, 3, 4)) / (
            self.params.patch_size * self.params.patch_size * 3
        ) < self.params.percentage_blank  # remove patch with only blanks pixels
        non_white_patches = output_patches[mask]

        indexes_to_sample = [
            random.randint(0, non_white_patches.shape[0] - 1)
            for _ in range(self.params.nb_samples)
        ]
        output_tensor = torch.stack(
            [non_white_patches[i] / 255.0 for i in indexes_to_sample]
        )

        if self.transform:
            output_tensor = self.transform(output_tensor)

        return output_tensor, label
