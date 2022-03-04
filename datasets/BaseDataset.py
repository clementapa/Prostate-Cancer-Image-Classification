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

        img_path = self.X[idx]
        label = self.y[idx]

        img = imread(img_path, key=2)

        # if self.transform !=None :
        #     img = self.transform(img)
        # im_scale = imread(path_im, key=2)

        im_tensor = torch.from_numpy(img)

        quantity_to_pad = abs(im_tensor.shape[0] - im_tensor.shape[1])
        bool_temp = im_tensor.shape[1] < im_tensor.shape[0]
        padded_im_tensor = F.pad(
            im_tensor,
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
            padded_im_tensor.shape[1] == padded_im_tensor.shape[2]
        )  # check that it is a square image

        remaining_pixels = padded_im_tensor.shape[1] % self.params.patch_size
        if remaining_pixels != 0:
            if (
                padded_im_tensor.shape[1] + remaining_pixels
            ) % self.params.patch_size == 0:
                # padd
                padded_im_tensor = F.pad(
                    padded_im_tensor,
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
                padded_im_tensor = padded_im_tensor[
                    :,
                    0 : padded_im_tensor.shape[1] - remaining_pixels,
                    0 : padded_im_tensor.shape[2] - remaining_pixels,
                    :,
                ]

        h = padded_im_tensor.shape[1] // self.params.patch_size
        w = padded_im_tensor.shape[2] // self.params.patch_size
        output_patches = rearrange(
            padded_im_tensor,
            "b (h p1) (w p2) c -> b (h w) p1 p2 c",
            p1=self.params.patch_size,
            p2=self.params.patch_size,
            h=h,
            w=w,
        )

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
