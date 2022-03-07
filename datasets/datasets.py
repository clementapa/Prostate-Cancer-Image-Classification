import os
import random
import torch
import openslide
import zipfile

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.dataset_utils import (
    parse_csv,
    parse_csv_static,
    return_random_patch,
)


class BaseDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, train=True, transform=None):
        super().__init__()

        self.params = params

        if train:
            self.X, self.y = parse_csv(params.root_dataset, "train")
        else:
            self.X, self.y = parse_csv(params.root_dataset, "test")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")

class BaseStaticDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, train=True, transform=None, wb_run=None):
        super().__init__()

        self.params = params
        
        # TODO wrap everything in a function

        if train:
            path_artifact = self.params.train_artifact.split('/')[-1].split(':')[0]
            if not os.path.exists(os.path.join(self.params.path_patches, path_artifact)):
                artifact = wb_run.use_artifact(self.params.train_artifact)
                datadir = artifact.download(root=self.params.path_patches)

                path_to_zip_file = os.path.join(self.params.path_patches, path_artifact+'.zip')
                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(datadir)
            self.X, self.y = parse_csv_static(params.root_dataset, "train", params.path_patches, path_artifact)
        else:
            path_artifact = self.params.test_artifact.split('/')[-1].split(':')[0]
            if not os.path.exists(os.path.join(self.params.path_patches, path_artifact)):
                artifact = wb_run.use_artifact(self.params.test_artifact)
                datadir = artifact.download(root=self.params.path_patches)

                path_to_zip_file = os.path.join(self.params.path_patches, path_artifact+'.zip')
                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(datadir)
            self.X, self.y = parse_csv_static(params.root_dataset, "test", params.path_patches, path_artifact)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")

class PatchDataset(BaseDataset):
    def __init__(self, params, train=True, transform=None):
        super().__init__(params, train, transform)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):
        img_path = self.X[idx]
        label = self.y[idx]

        wsi_image = openslide.OpenSlide(img_path)
        pil_imgs = [
            return_random_patch(
                wsi_image, self.params.patch_size, self.params.percentage_blank
            )
            for _ in range(self.params.nb_samples)
        ]
        output_tensor = torch.stack([self.transform(pil_img) for pil_img in pil_imgs])

        return output_tensor, label

class StaticPatchDataset(BaseStaticDataset):
    def __init__(self, params, train=True, transform=None, wb_run=None):
        super().__init__(params, train, transform, wb_run)
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


    def __getitem__(self, idx):
        np_path = self.X[idx]
        np_array = np.load(open(np_path, "rb"))
        
        label = self.y[idx]
      
        output_tensor = torch.stack([self.transform((torch.from_numpy(np_img)/255.0).permute(2,1,0)) for np_img in np_array[:self.params.nb_samples]])

        return output_tensor, label

