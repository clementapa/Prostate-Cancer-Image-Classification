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


class MMSg3(nn.Module):
    def __init__(self, params, wb_run=None):
        super().__init__()
        self.params = params

        in_shape = params.nb_samples * 3

        self.classifier = nn.Sequential(
                                nn.Linear(in_shape, in_shape * 2),
                                getattr(nn, params.activation)(),
                                nn.Linear(in_shape*2, in_shape * 4),
                                getattr(nn, params.activation)(),
                                nn.Linear(in_shape*4, in_shape * 8),
                                getattr(nn, params.activation)(),
                                nn.Linear(in_shape*8, in_shape * 16),
                                getattr(nn, params.activation)(),
                                nn.Linear(in_shape*16, in_shape * 8),
                                getattr(nn, params.activation)(),
                                nn.Linear(in_shape*8, 6),
                            )

        # load seg_model
        name_artifact = f"attributes_classification_celeba/test-dlmi/{params.wb_run_seg}:top-1"
        artifact = wb_run.use_artifact(name_artifact)
        path_to_model = artifact.download()
        # path_to_model = "/home/younesbelkada/Travail/MVA/DeepMedical/Prostate-Cancer-Image-Classification/artifacts/expert-surf-171:v7/epoch-19-val_loss=0.17.ckpt"

        base_module = BaseModuleForInference(params)
        base_module.load_state_dict(torch.load(os.path.join(path_to_model, os.listdir(path_to_model)[0]), map_location=torch.device('cpu'))['state_dict'])
        # base_module.load_state_dict(torch.load(os.path.join(path_to_model, os.listdir(path_to_model)[0]))['state_dict'])
        self.seg_model = base_module.model

    def forward(self, x):
        scores = []
        
        for batch_patch in x:
        
            with torch.no_grad():
                seg_mask = self.seg_model(batch_patch).argmax(dim=1)

            seg_score = seg_max_to_score(seg_mask, seg_mask.shape[-1])

            scores.append(seg_score)

        scores = torch.stack(scores)     
        scores = rearrange(scores, "b p d -> b (p d)")

        output = self.classifier(scores)
        
        return output