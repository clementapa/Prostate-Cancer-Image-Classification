import csv
import os
import torch


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
