import os
import timm
import torch
import torch.nn as nn

from einops import rearrange
from models.BaseModule import BaseModuleForInference
from models.Segmentation import Segmentation

from models.MLP import MLP
from utils.agent_utils import get_artifact
from utils.dataset_utils import seg_max_to_score


class MSgScore(nn.Module):
    def __init__(self, params, wb_run=None):
        super().__init__()
        self.params = params

        self.mlp = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 6))

        # load seg_model
        name_artifact = (
            f"attributes_classification_celeba/test-dlmi/{params.wb_run_seg}:top-1"
        )
        artifact = wb_run.use_artifact(name_artifact)
        path_to_model = artifact.download()
        # path_to_model = "/home/younesbelkada/Travail/MVA/DeepMedical/Prostate-Cancer-Image-Classification/artifacts/expert-surf-171:v7/epoch-19-val_loss=0.17.ckpt"

        base_module = BaseModuleForInference(params)
        # base_module.load_state_dict(torch.load(os.path.join(path_to_model, os.listdir(path_to_model)[0]), map_location=torch.device('cpu'))['state_dict'])
        base_module.load_state_dict(
            torch.load(os.path.join(path_to_model, os.listdir(path_to_model)[0]))[
                "state_dict"
            ]
        )
        self.seg_model = base_module.model
        # self.seg_model._requires_grad(False)
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        scores = []
        with torch.no_grad():
            for batch in x:
                seg_mask = self.seg_model(batch).argmax(dim=1)
                score = seg_max_to_score(seg_mask, seg_mask.shape[-1])
                scores.append(score)
        # bs, n_patches, h, w, c
        # features = torch.stack(features)
        scores = torch.stack(scores).mean(dim=1)
        output = self.mlp(scores)
        return output
