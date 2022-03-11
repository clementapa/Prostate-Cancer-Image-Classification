import argparse

# import json
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np

# import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

# from patchify import patchify
from tifffile import imread
from torchvision.utils import make_grid
from tqdm import tqdm
from config.hparams import Parameters
from torchvision.transforms import transforms

import wandb


from models.BaseModule import BaseModuleForInference
from utils.dataset_utils import seg_max_to_score


def main(params, wb_run_seg, patch_size, split, percentage_blank, level):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wb_run = wandb.init(entity="attributes_classification_celeba", project="dlmi")
    name_artifact = f"attributes_classification_celeba/test-dlmi/{wb_run_seg}:top-1"
    artifact = wb_run.use_artifact(name_artifact)
    path_to_model = artifact.download()

    base_module = BaseModuleForInference(params)
    base_module.load_state_dict(torch.load(os.path.join(path_to_model, os.listdir(path_to_model)[0]))['state_dict'])
    seg_model = base_module.model.to(device)

    transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomHorizontalFlip(),
                ]
            )

    root_dataset = osp.join(os.getcwd(), "assets", "mvadlmi")

    df = pd.read_csv(osp.join(root_dataset, split + ".csv"))

    patch_path = osp.join(
        os.getcwd(),
        "assets",
        "dataset_patches_seg",
        f"{split}_{patch_size}_{level}_{percentage_blank}",
    )

    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    for i in tqdm(df.index):
        img_path = osp.join("assets",  "dataset_patches", '_'.join([split, str(patch_size), str(1), str(percentage_blank)]), df["image_id"][i] + ".npy")
        np_array = np.load(open(img_path, "rb"))
        non_white_patches = torch.from_numpy(np_array)
        with torch.no_grad():
            seg_masks_batch_1 = seg_model(transform(non_white_patches.permute(0, 3, 2, 1)/255.0).to(device)[:16]).argmax(dim=1)
            seg_masks_batch_2 = seg_model(transform(non_white_patches.permute(0, 3, 2, 1)/255.0).to(device)[16:]).argmax(dim=1)
            
            scores = [seg_max_to_score(seg_masks_batch_1, patch_size).cpu().numpy(), seg_max_to_score(seg_masks_batch_2, patch_size).cpu().numpy()]
            seg_scores = np.concatenate(scores, axis=0)
            np.save(osp.join(patch_path, df["image_id"][i]), np.concatenate([seg_masks_batch_1.cpu().numpy(), seg_masks_batch_2.cpu().numpy()], axis=0))
            np.save(osp.join(patch_path, df["image_id"][i]+'_score'), seg_scores)
        

    zip_name = osp.join(
        osp.join(
            os.getcwd(),
            "assets",
            "dataset_patches_seg",
            f"{split}_{patch_size}_{level}_{percentage_blank}",
        )
    )
    shutil.make_archive(zip_name, "zip", patch_path)

    # Push artifact

    artifact = wandb.Artifact(
        name=os.path.basename(f"{zip_name}_score"),
        type="dataset",
        metadata={
            "split": split,
            "patch_size": patch_size,
            "percentage_blank": percentage_blank,
            "level": level,
        },
        description=f" {split} dataset of masks split by patches {patch_size}, level {level}, percentage_blank {percentage_blank} together with their score vectors",
    )

    artifact.add_file(zip_name + ".zip")
    wandb.log_artifact(artifact, aliases=["latest"])


if __name__ == "__main__":
    params = Parameters.parse()

    split = params.push_artifact_params.split
    patch_size = params.data_param.patch_size
    percentage_blank = params.data_param.percentage_blank
    level = params.push_artifact_params.level
    wb_run_seg = params.network_param.wb_run_seg

    main(params.network_param, wb_run_seg, patch_size, split, percentage_blank, level)