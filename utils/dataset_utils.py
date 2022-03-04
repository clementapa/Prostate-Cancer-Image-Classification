import csv
import os, random
import torch
import openslide
import numpy as np


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


# def coll_fn(batch):
#     N = min([b[0].shape[-1] for b in batch])
#     y = torch.LongTensor([b[1] for b in batch])

#     X = torch.stack([b[0][:N] for b in batch])
#     return X, y


def coll_fn(batch):
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
