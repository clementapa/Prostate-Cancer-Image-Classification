import numpy as np
import cv2

from datasets.datamodule import BaseDataModule
from utils.utils_dataset import parse_csv

class BaseDataset(BaseDataModule):
    """
        A base dataset module to load the dataset for the challenge
    """
    def __init__(self, params):
        super().__init__(params)
        self.X, self.y = parse_csv(params.root_dataset)
        self.X_train, self.y_test = parse_csv(params.root_dataset, 'test')
    
    def __getitem__(self, idx):
        pass
