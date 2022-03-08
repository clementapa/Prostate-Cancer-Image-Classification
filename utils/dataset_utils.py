import random

import albumentations as albu
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2


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
    while (
        np.sum(
            np.any(np.array(cropped_image)[:, :, :-1] == [255.0, 255.0, 255.0], axis=-1)
        )
        > percentage_blank * patch_dim * patch_dim
    ):
        random_location_x = random.randint(0, wsi_dimensions[0] - patch_dim)
        random_location_y = random.randint(0, wsi_dimensions[1] - patch_dim)
        cropped_image = whole_slide.read_region(
            (random_location_x, random_location_y), level, (patch_dim, patch_dim)
        )
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

def seg_max_to_score(seg_mask):
    score_0 = torch.where(seg_mask==0, 1.0, 0.0).sum(axis=-1).sum(axis=-1)
    score_1 = torch.where(seg_mask==1, 1.0, 0.0).sum(axis=-1).sum(axis=-1)
    score_2 = torch.where(seg_mask==2, 1.0, 0.0).sum(axis=-1).sum(axis=-1)
    return torch.stack([score_0, score_1, score_2], dim=-1)/(256*256)