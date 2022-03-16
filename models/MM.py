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


class MM(nn.Module):
    def __init__(self, params, wb_run=None):
        super().__init__()
        self.params = params

        self.norm = getattr(nn, params.normalization)
        self.activation = getattr(nn, params.activation)

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )
        self.features_extractor.reset_classifier(0)
        in_shape = self.features_extractor(torch.randn(1, 3, 224, 224)).shape[1]

        # self.patch_selector = nn.Sequential(
        #     nn.Linear(in_shape, 1),
        #     nn.Sigmoid()
        # )

        self.patch_selector = nn.Sequential(
            nn.Linear(in_shape, in_shape),
            self.norm(in_shape),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(in_shape, in_shape // 2),
            self.norm(in_shape // 2),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(in_shape // 2, 1),
            nn.Sigmoid(),
        )

        self.norm = getattr(nn, params.normalization)
        self.activation = getattr(nn, params.activation)

        # self.mlp = MLP(params.bottleneck_shape * params.nb_samples, params)
        if self.params.classifier_name == "MLP":
            self.classifier = MLP(in_shape + 1, params)
        elif self.params.classifier_name == "Linear":
            self.classifier = nn.Linear(in_shape + 1, 6)
        elif self.params.classifier_name == "Multiple Linear":
            self.classifier = nn.Sequential(
                # nn.Linear(in_shape+4, in_shape+4),
                # self.norm(in_shape+4),
                # self.activation(),
                # nn.Dropout(params.dropout),
                nn.Linear(in_shape + 1, (in_shape + 1) // 2),
                # self.norm((in_shape+4)//2),
                self.activation(),
                nn.Dropout(params.dropout),
                nn.Linear((in_shape + 1) // 2, 6),
            )
        else:
            raise NotImplementedError("Classifier not implemented ! MLP or Linear")

    def forward(self, x):
        patch_scores = []
        features = []

        for batch in x:
            feature = self.features_extractor(batch)

            patch_score = self.patch_selector(feature)

            patch_scores.append(patch_score)
            features.append(feature)

        most_relevant_patch = torch.argmax(torch.stack(patch_scores), dim=1)

        features = torch.stack(features)
        patch_scores = torch.stack(patch_scores)

        features_important_patches = features[
            torch.arange(x.size(0)), most_relevant_patch.squeeze()
        ]
        probas = patch_scores[
            torch.arange(patch_scores.size(0)), most_relevant_patch.squeeze()
        ]

        output = self.classifier(
            torch.cat([features_important_patches, probas], dim=-1)
        )

        return (output, probas)
