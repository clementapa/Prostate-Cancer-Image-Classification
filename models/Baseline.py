import timm
import torch
import torch.nn as nn

from models.MLP import MLP


class Baseline(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )  # TODO how deal with the input?

        self.mlp = MLP(self.features_extractor.fc.in_features, params)

        self.features_extractor.fc = nn.Identity()

    def forward(self, x):
        # step 1: random sampling patches
        features = []
        for batch in x:
            features.append(self.features_extractor(batch.permute(0, 3, 1, 2)))
        # bs, n_patches, h, w, c
        features = torch.stack(features)

        output = self.mlp(features.mean(dim=1))  # mean to try
        return output
