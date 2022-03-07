import argparse

# import json
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np

# import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

# from patchify import patchify
from tifffile import imread
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb


def main(patch_size, split, percentage_blank, level):

    root_dataset = osp.join(os.getcwd(), "assets", "mvadlmi")

    df = pd.read_csv(osp.join(root_dataset, split + ".csv"))

    patch_path = osp.join(
        os.getcwd(),
        "assets",
        "dataset_patches",
        f"{split}_{patch_size}_{level}_{percentage_blank}",
    )

    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    for i in tqdm(df.index):

        img_path = osp.join(root_dataset, split, split, df["image_id"][i] + ".tiff")

        # wsi_image = openslide.OpenSlide(img_path)

        # wsi_dimensions = wsi_image.level_dimensions

        # img = wsi_image.read_region(
        #     wsi_dimensions[level], level, wsi_dimensions[level]
        # )
        img = imread(img_path, key=level)
        # plt.imshow(img)
        # plt.show()
        img = torch.from_numpy(img)

        # img_patched = patchify(img, (patch_size, patch_size), step=1)

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

        assert img.shape[1] == img.shape[2]  # check that it is a square image

        # process image to divide per patch
        remaining_pixels = img.shape[1] % patch_size
        if remaining_pixels != 0:
            if (img.shape[1] + remaining_pixels) % patch_size == 0:
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
        h = img.shape[1] // patch_size
        w = img.shape[2] // patch_size
        img = rearrange(
            img,
            "b (h p1) (w p2) c -> b (h w) p1 p2 c",
            p1=patch_size,
            p2=patch_size,
            h=h,
            w=w,
        )
        # Remove white patches
        mask = (1.0 * (img >= 240)).sum(dim=(2, 3, 4)) / (
            patch_size * patch_size * 3
        ) <= percentage_blank  # remove patch with only blanks pixels
        non_white_patches = img[mask]
        
        np.save(osp.join(patch_path, df["image_id"][i]), non_white_patches.numpy())

    zip_name = osp.join(
        osp.join(
            os.getcwd(),
            "assets",
            "dataset_patches",
            f"{split}_{patch_size}_{level}_{percentage_blank}",
        )
    )
    shutil.make_archive(zip_name, "zip", patch_path)

    # Push artifact
    wandb.init(entity="attributes_classification_celeba", project="dlmi")

    artifact = wandb.Artifact(
        name=os.path.basename(zip_name),
        type="dataset",
        metadata={
            "split": split,
            "patch_size": patch_size,
            "percentage_blank": percentage_blank,
            "level": level,
        },
        description=f" {split} dataset of images split by patches {patch_size}, level {level}, percentage_blank {percentage_blank} ",
    )

    artifact.add_file(zip_name + ".zip")
    wandb.log_artifact(artifact, aliases=["latest"])  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push an artifact to wandb")
    parser.add_argument("--patch_size", required=True, type = int, help = "patch size ")
    parser.add_argument("--split", required=True, type = str, help = "name of the split")
    parser.add_argument("--percentage_blank", required=False, type = float, help = "percentage of blank pixels", default=0.5)
    parser.add_argument("--level", required=False, type = int, help = "Level of tiff file", default=1)
    args = parser.parse_args()
    
    split = args.split
    patch_size = args.patch_size
    percentage_blank = args.percentage_blank
    level = args.level
    
    main(patch_size, split, percentage_blank, level)
