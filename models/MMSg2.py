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


class MMSg2(nn.Module):
    def __init__(self, params, wb_run=None):
        super().__init__()
        self.params = params

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )
        self.features_extractor.reset_classifier(0)
        in_shape = self.features_extractor(torch.randn(1, 3, 224, 224)).shape[1]

        # self.mlp = MLP(params.bottleneck_shape * params.nb_samples, params)
        self.mlp = MLP(in_shape, params)

        # load seg_model
        name_artifact = f"attributes_classification_celeba/test-dlmi/{params.wb_run_seg}:top-1"
        artifact = wb_run.use_artifact(name_artifact)
        path_to_model = artifact.download()
        # path_to_model = "/home/younesbelkada/Travail/MVA/DeepMedical/Prostate-Cancer-Image-Classification/artifacts/expert-surf-171:v7/epoch-19-val_loss=0.17.ckpt"

        base_module = BaseModuleForInference(params)
        # base_module.load_state_dict(torch.load(os.path.join(path_to_model, os.listdir(path_to_model)[0]), map_location=torch.device('cpu'))['state_dict'])
        base_module.load_state_dict(torch.load(os.path.join(path_to_model, os.listdir(path_to_model)[0]))['state_dict'])
        self.seg_model = base_module.model
        for param in self.seg_model.parameters():
            param.requires_grad = False
        # self.seg_model._requires_grad(False)

    def forward(self, x):
        top_k_patches = []
        scores = []
        with torch.no_grad():
            for batch in x:
                seg_mask = self.seg_model(batch).argmax(dim=1)
                score = seg_max_to_score(seg_mask, seg_mask.shape[-1])
                top_k_patches.append(torch.topk(score[:, -1], x.shape[1]//2, dim=-1).indices)
                scores.append(score)
            # most_relevant_patch = torch.argmax(torch.stack(scores), dim=1)
        # bs, n_patches, h, w, c
        # features = torch.stack(features)
        top_k_patches = torch.stack(top_k_patches)
        filtered_features = []

        for i, top_k in enumerate(top_k_patches):
            selected_patches = x[i, top_k]
            features = self.features_extractor(selected_patches)
            filtered_features.append(features)
        mean_filtered_features = torch.stack(filtered_features).mean(dim=1)
        output = self.mlp(mean_filtered_features)
        return output