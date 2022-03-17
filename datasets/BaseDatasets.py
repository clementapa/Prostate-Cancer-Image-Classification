import os
import os.path as osp
import zipfile

import numpy as np
import torchvision.transforms as transforms
import wandb

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, X, y, df, train=True, static=False):
        super().__init__()

        self.params = params
        self.train = train

        if self.train:
            self.subpath = osp.join("train", "train")
        else:
            self.subpath = osp.join("test", "test")

        if static:
            if train:
                self.name_dataset = osp.basename(self.params.train_artifact).split(":")[
                    0
                ]
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

            else:
                self.name_dataset = osp.basename(self.params.test_artifact).split(":")[
                    0
                ]
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
