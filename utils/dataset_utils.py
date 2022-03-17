import os
import os.path as osp
import random
import shutil

import albumentations as albu
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from albumentations.pytorch.transforms import ToTensorV2
from einops import rearrange
from PIL import Image
from tifffile import imread
from tqdm import tqdm

def merge_cls(seg_img):
    seg_img[seg_img == 2] = 1
    seg_img[(seg_img == 7)] = 0
    seg_img[(seg_img >= 3)] = 2
    return seg_img


def coll_fn(batch):
    y = torch.LongTensor([b[1] for b in batch])
    X = torch.stack([b[0] for b in batch])
    return X, y


def coll_fn_seg(batch):
    y = torch.stack([b[1] for b in batch])
    X = torch.stack([b[0] for b in batch])

    X = torch.cat(X.unbind())
    y = torch.cat(y.unbind())
    return X, y


def return_random_patch(whole_slide, patch_dim, percentage_blank, level):
    wsi_dimensions = whole_slide.dimensions
    random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
    random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)
    cropped_image = whole_slide.read_region(
        (random_location_x, random_location_y), level, (patch_dim, patch_dim)
    )

    # Another preprocessing to remove the black pixels...

    cropped_image_array = np.array(cropped_image)[:, :, :-1]
    mask = np.all(cropped_image_array == [0, 0, 0], axis=2)
    cropped_image_array[mask] = [255.0, 255.0, 255.0]
    cropped_image = Image.fromarray(cropped_image_array)

    while (
        np.sum(np.any(np.array(cropped_image) == [255.0, 255.0, 255.0], axis=-1))
        > percentage_blank * patch_dim * patch_dim
    ):
        random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
        random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)
        cropped_image = whole_slide.read_region(
            (random_location_x, random_location_y), level, (patch_dim, patch_dim)
        )

        cropped_image_array = np.array(cropped_image)[:, :, :-1]
        mask = np.all(cropped_image_array == [0, 0, 0], axis=2)
        cropped_image_array[mask] = [255.0, 255.0, 255.0]
        cropped_image = Image.fromarray(cropped_image_array)

    return cropped_image.convert("RGB")


def return_random_patch_with_mask(
    whole_slide, seg_slide, patch_dim, percentage_blank, level
):
    wsi_dimensions = whole_slide.dimensions
    random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
    random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)
    cropped_mask = seg_slide.read_region(
        (random_location_x, random_location_y), level, (patch_dim, patch_dim)
    )
    while (
        np.sum(1.0 * (np.array(cropped_mask)[:, :, 0] == 0))
        > percentage_blank * patch_dim * patch_dim
    ):
        random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
        random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)

        cropped_mask = seg_slide.read_region(
            (random_location_x, random_location_y), level, (patch_dim, patch_dim)
        )

    cropped_image = whole_slide.read_region(
        (random_location_x, random_location_y), level, (patch_dim, patch_dim)
    )

    assert (np.max(np.array(cropped_mask.convert("RGB"))[:, :, 1]) == 0) and (
        np.max(np.array(cropped_mask.convert("RGB"))[:, :, 2]) == 0
    )

    cropped_mask = torch.as_tensor(
        np.array(cropped_mask.convert("RGB"))[:, :, 0], dtype=torch.int64
    )

    return cropped_image.convert("RGB"), cropped_mask


def get_training_augmentation(num_classes):
    train_transform = [albu.HorizontalFlip(p=0.5), ToTensorV2()]

    return albu.Compose(train_transform)


def get_validation_augmentation(num_classes):
    test_transform = [ToTensorV2()]
    return albu.Compose(test_transform)


def seg_max_to_score(seg_mask, patch_size):
    score_0 = torch.where(seg_mask == 0, 1.0, 0.0).sum(axis=-1).sum(axis=-1)
    score_1 = torch.where(seg_mask == 1, 1.0, 0.0).sum(axis=-1).sum(axis=-1)
    score_2 = torch.where(seg_mask == 2, 1.0, 0.0).sum(axis=-1).sum(axis=-1)
    return torch.stack([score_0, score_1, score_2], dim=-1) / (patch_size * patch_size)


def analyse_repartition(train_dataset, val_dataset):

    plot_split("train", train_dataset)
    plot_split("val", val_dataset)


def _count(array):
    unique, counts = np.unique(array, return_counts=True)
    count = dict(zip(unique, counts))
    count = [[label, val] for (label, val) in count.items()]
    return count


def plot_split(name_split, dataset):

    targets = dataset.get_targets()
    providers = dataset.get_providers()
    targets_count = _count(targets)
    providers_count = _count(providers)

    table = wandb.Table(data=targets_count, columns=["Class", "Count"])
    wandb.log(
        {
            f"Data Analysis/{name_split}_classes": wandb.plot.bar(
                table, "Class", "Count", title=f"Classes repartition {name_split}"
            )
        }
    )

    table = wandb.Table(data=providers_count, columns=["Data provider", "Count"])
    wandb.log(
        {
            f"Data Analysis/{name_split}_providers": wandb.plot.bar(
                table,
                "Data provider",
                "Count",
                title=f"Data providers repartition {name_split}",
            )
        }
    )

def images_to_patches(root_dataset, patch_size, split, percentage_blank, level):
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

        img = imread(img_path, key=level)
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

    return patch_path