from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

import os, os.path as osp
import pandas as pd
import numpy as np

from utils.dataset_utils import coll_fn, coll_fn_seg, analyse_repartition
import datasets.datasets as datasets


class BaseDataModule(LightningDataModule):
    def __init__(self, mode, dataset_param):
        super().__init__()

        self.config = dataset_param
        self.batch_size = self.config.batch_size
        self.mode = mode

        if self.mode == "Segmentation":
            self.collate_fn = coll_fn_seg
        else:
            self.collate_fn = coll_fn

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):
            # Load dataset

            df = pd.read_csv(osp.join(self.config.root_dataset, "train.csv"))
            
            if self.mode == "Segmentation":
                name = df["image_id"] + ".tiff"
                mask = name.isin(
                    os.listdir(osp.join(self.config.root_dataset, "train_label_masks", "train_label_masks"))
                )
                df = df[mask].copy()
            
            X = np.array(df['image_id'])
            y = np.array(df['isup_grade'])

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.config.split_val, stratify=y)

            self.train_dataset = getattr(datasets, self.config.dataset_name)(
                self.config, X_train, y_train, df, train=True
            )
            self.val_dataset = getattr(datasets, self.config.dataset_name)(
                self.config, X_val, y_val, df, train=True
            )

            analyse_repartition(self.train_dataset, self.val_dataset)

        if stage == "predict":

            df = pd.read_csv(osp.join(self.config.root_dataset, "test.csv"))
            X_test = np.array(df['image_id'])
            y_dummy = np.ones_like(X_test)

            self.dataset = getattr(datasets, self.config.dataset_name)(
                self.config, X_test, y_dummy, train=False
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        return val_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        return predict_loader
