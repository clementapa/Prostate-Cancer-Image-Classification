import os.path as osp

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, X, y, df, train, static):
        super().__init__()

        self.params = params
        self.train = train

        if self.train:
            self.subpath = osp.join("train", "train")
        else:
            self.subpath = osp.join("test", "test")

        self.df = df
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def get_targets(self):
        return np.array(self.df[self.df["image_id"].isin(self.X)]["isup_grade"])

    def get_providers(self):
        return np.array(self.df[self.df["image_id"].isin(self.X)]["data_provider"])

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")
