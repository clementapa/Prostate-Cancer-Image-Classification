import torch
import openslide

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.dataset_utils import (
    merge_cls,
    parse_csv,
    parse_csv_seg,
    return_random_patch,
    get_segmentation_paths,
    return_random_patch_with_mask,
)


class BaseDataset(Dataset):
    """
    A base dataset module to load the dataset for the challenge
    """

    def __init__(self, params, train=True, transform=None, segmentation=False):
        super().__init__()

        self.params = params

        if train:
            if segmentation:
                self.X, self.y = parse_csv_seg(
                    params.root_dataset, "train", params.data_provider
                )
                self.X, self.path_seg, self.y = get_segmentation_paths(self.X, self.y)
            else:
                self.X, self.y = parse_csv(params.root_dataset, "train")
        else: # TODO to fix when inferring
            if segmentation:
                self.X, self.y = parse_csv_seg(
                    params.root_dataset, "test", params.data_provider
                )
            else:
                self.X, self.y = parse_csv(params.root_dataset, "test")

        if transform == None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError(f"Should be implemented in derived class!")


class PatchDataset(BaseDataset):
    def __init__(self, params, train=True, transform=None):
        super().__init__(params, train, transform, segmentation=False)

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


class PatchSegDataset(BaseDataset):
    def __init__(self, params, train=True, transform=None):
        super().__init__(params, train, transform, segmentation=True)

    def __getitem__(self, idx):
        img_path = self.X[idx]
        seg_path = self.path_seg[idx]

        wsi_image = openslide.OpenSlide(img_path)
        wsi_seg = openslide.OpenSlide(seg_path)

        pil_imgs = []
        seg_gt = []
        
        for _ in range(self.params.nb_samples):
            pil_img, seg_img = return_random_patch_with_mask(
                wsi_image, wsi_seg, self.params.patch_size, self.params.percentage_blank
            )

            if self.params.data_provider == "radboud_merged":
                seg_img = merge_cls(seg_img)
            pil_imgs.append(pil_img)
            seg_gt.append(seg_img)
        output_tensor = torch.stack([self.transform(pil_img) for pil_img in pil_imgs])
        seg_masks = torch.stack(seg_gt)
        
        return output_tensor, seg_masks