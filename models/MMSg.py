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


class MMSg(nn.Module):
    def __init__(self, params, wb_run=None):
        super().__init__()
        self.params = params

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )
        self.features_extractor.reset_classifier(0)
        in_shape = self.features_extractor(torch.randn(1, 3, 224, 224)).shape[1]

        self.patch_selector = nn.Sequential(nn.Linear(in_shape + 3, 1), nn.Sigmoid())

        # self.mlp = MLP(params.bottleneck_shape * params.nb_samples, params)
        if self.params.classifier_name == "MLP":
            self.classifier = MLP(in_shape + 4, params)
        elif self.params.classifier_name == "Linear":
            self.classifier = nn.Linear(in_shape + 4, 6)
        elif self.params.classifier_name == "Multiple Linear":
            self.classifier = nn.Sequential(
                nn.Linear(in_shape + 4, (in_shape + 4) // 2),
                getattr(nn, params.activation)(),
                nn.Linear((in_shape + 4) // 2, 6),
            )
        else:
            raise NotImplementedError("Classifier not implemented ! MLP or Linear")

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

    def forward(self, x):
        patch_scores = []
        scores = []
        features = []

        for batch in x:
            feature = self.features_extractor(batch)

            with torch.no_grad():
                seg_mask = self.seg_model(batch).argmax(dim=1)

            seg_score = seg_max_to_score(seg_mask, seg_mask.shape[-1])
            patch_score = self.patch_selector(torch.cat([feature, seg_score], dim=-1))

            patch_scores.append(patch_score)
            scores.append(seg_score)
            features.append(feature)

        most_relevant_patch = torch.argmax(torch.stack(patch_scores), dim=1)

        scores = torch.stack(scores)
        features = torch.stack(features)
        patch_scores = torch.stack(patch_scores)

        features_important_patches = features[
            torch.arange(scores.size(0)), most_relevant_patch.squeeze()
        ]
        scores_important_patches = scores[
            torch.arange(scores.size(0)), most_relevant_patch.squeeze()
        ]
        probas = patch_scores[
            torch.arange(patch_scores.size(0)), most_relevant_patch.squeeze()
        ]

        output = self.classifier(
            torch.cat(
                [features_important_patches, scores_important_patches, probas], dim=-1
            )
        )

        return (output, probas)
