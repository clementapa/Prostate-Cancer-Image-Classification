import csv
import os
import random
from einops import rearrange

import albumentations as albu
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image

from patchify import patchify

from albumentations.pytorch.transforms import ToTensorV2


def merge_cls(seg_img):
    seg_img[seg_img == 2] = 1
    seg_img[(seg_img == 3)] = 2
    seg_img[(seg_img == 4)] = 2
    seg_img[(seg_img == 5)] = 2
    return seg_img


def parse_csv(path_dataset, split="train"):
    X = []
    y = []
    path_images = os.path.join(path_dataset, split, split)
    path_csv_file = os.path.join(path_dataset, split + ".csv")
    with open(path_csv_file, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for i, row in enumerate(spamreader):
            if i > 0:
                X.append(os.path.join(path_images, row[0] + ".tiff"))
                y.append(int(row[-2]))
    return X, y


def parse_csv_seg(path_dataset, split="train", data_provider="radboud"):
    X = []
    y = []
    data_provider = data_provider.replace("_merged", "")
    path_images = os.path.join(path_dataset, split, split)
    path_csv_file = os.path.join(path_dataset, split + ".csv")
    with open(path_csv_file, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for i, row in enumerate(spamreader):
            if i > 0 and row[1] == data_provider:
                X.append(os.path.join(path_images, row[0] + ".tiff"))
                y.append(int(row[-2]))
    return X, y


def get_segmentation_paths(input_paths, labels):
    segmentation_paths = []
    cleaned_input_paths = []
    cleaned_labels = []
    for i, path in enumerate(input_paths):
        if os.path.exists(path.replace("train", "train_label_masks")):
            segmentation_paths.append(path.replace("train", "train_label_masks"))
            cleaned_input_paths.append(path)
            cleaned_labels.append(labels[i])
    return cleaned_input_paths, segmentation_paths, cleaned_labels


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


def coll_fn_(batch):
    y = torch.LongTensor([b[1] for b in batch])
    X = torch.stack([b[0] for b in batch])

    return X, y


def return_random_patch(whole_slide, patch_dim, percentage_blank):
    wsi_dimensions = whole_slide.dimensions
    random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
    random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)
    cropped_image = whole_slide.read_region(
        (random_location_x, random_location_y), 0, (patch_dim, patch_dim)
    )
    while (
        np.sum(
            np.any(np.array(cropped_image)[:, :, :-1] == [255.0, 255.0, 255.0], axis=-1)
        )
        > percentage_blank * patch_dim * patch_dim
    ):
        random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
        random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)
        cropped_image = whole_slide.read_region(
            (random_location_x, random_location_y), 0, (patch_dim, patch_dim)
        )
    return cropped_image.convert("RGB")


def return_random_patch_with_mask(whole_slide, seg_slide, patch_dim, percentage_blank):
    wsi_dimensions = whole_slide.dimensions
    random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
    random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)
    cropped_mask = seg_slide.read_region(
        (random_location_x, random_location_y), 0, (patch_dim, patch_dim)
    )
    while (
        np.sum(1.0 * (np.array(cropped_mask)[:, :, 0] == 0))
        > percentage_blank * patch_dim * patch_dim
    ):
        random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
        random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)

        cropped_mask = seg_slide.read_region(
            (random_location_x, random_location_y), 0, (patch_dim, patch_dim)
        )

    cropped_image = whole_slide.read_region(
        (random_location_x, random_location_y), 0, (patch_dim, patch_dim)
    )

    assert (np.max(np.array(cropped_mask.convert("RGB"))[:, :, 1]) == 0) and (
        np.max(np.array(cropped_mask.convert("RGB"))[:, :, 2]) == 0
    )

    cropped_mask = torch.as_tensor(
        np.array(cropped_mask.convert("RGB"))[:, :, 0], dtype=torch.int64
    )

    return cropped_image.convert("RGB"), cropped_mask


def image_to_patches(whole_slide, patch_size, percentage_blank):
    wsi_dimensions = whole_slide.dimensions
    img_patched = []

    whole_image = whole_slide.read_region(
        wsi_dimensions, 0, wsi_dimensions
    )
    whole_image = ToTensor()(whole_image)
    img_patched = patchify(whole_image, (patch_size, patch_size), step=1)
    # # Square image
    # quantity_to_pad = abs(img.shape[0] - img.shape[1])
    # bool_temp = img.shape[1] < img.shape[0]
    # img = F.pad(
    #     img,
    #     pad=(
    #         0,
    #         0,
    #         quantity_to_pad * bool_temp,
    #         0,
    #         quantity_to_pad * (1 - bool_temp),
    #         0,
    #     ),
    #     mode="constant",
    #     value=255,
    # ).unsqueeze(0)

    # assert img.shape[1] == img.shape[2]  # check that it is a square image

    # # process image to divide per patch
    # remaining_pixels = img.shape[1] % patch_size
    # if remaining_pixels != 0:
    #     if (img.shape[1] + remaining_pixels) % patch_size == 0:
    #         # padd
    #         img = F.pad(
    #             img,
    #             pad=(
    #                 0,
    #                 0,
    #                 remaining_pixels // 2,
    #                 remaining_pixels // 2,
    #                 remaining_pixels // 2,
    #                 remaining_pixels // 2,
    #             ),
    #             mode="constant",
    #             value=255,
    #         )
    #     else:
    #         # crop
    #         img = img[
    #             :,
    #             0 : img.shape[1] - remaining_pixels,
    #             0 : img.shape[2] - remaining_pixels,
    #             :,
    #         ]

    # # Divide image per patch
    # h = img.shape[1] // patch_size
    # w = img.shape[2] // patch_size
    # output_patches = rearrange(
    #     img,
    #     "b (h p1) (w p2) c -> b (h w) p1 p2 c",
    #     p1=patch_size,
    #     p2=patch_size,
    #     h=h,
    #     w=w,
    # )

    # Remove white patches
    # mask = (1.0 * (output_patches == 255)).sum(dim=(2, 3, 4)) / (
    #     patch_size * patch_size * 3
    # ) < percentage_blank  # remove patch with only blanks pixels
    # non_white_patches = output_patches[mask]

    return img_patched


def get_training_augmentation(num_classes):
    train_transform = [albu.HorizontalFlip(p=0.5), ToTensorV2()]

    return albu.Compose(train_transform)


def get_validation_augmentation(num_classes):
    test_transform = [ToTensorV2()]
    return albu.Compose(test_transform)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result