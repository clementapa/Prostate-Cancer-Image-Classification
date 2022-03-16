import timm
import torch
import torch.nn as nn

from einops import rearrange

from models.classifier.MLP import MLP
from models.BaseModels import BaseTopPatchMethods
from utils.agent_utils import get_features_extractor


class Baseline(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # get features extractor
        self.features_extractor, self.feature_size = get_features_extractor(
            params.feature_extractor_name
        )

        self.mlp = MLP(self.feature_size * params.nb_samples, params)

    def forward(self, x):
        features = []
        for batch in x:
            features.append(self.features_extractor(batch))

        # bs, n_patches, h, w, c
        features = torch.stack(features)

        output = self.mlp(rearrange(features, "b p d -> b (p d)"))
        return output


class TopPatch(BaseTopPatchMethods):
    def __init__(self, params):
        super().__init__(self, params)
        self.params = params

        self.patch_selector = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            self.norm(self.feature_size),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(self.feature_size, self.feature_size // 2),
            self.norm(self.feature_size // 2),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(self.feature_size // 2, 1),
            nn.Sigmoid(),
        )

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


class SimpleModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # get features extractor
        self.features_extractor, self.feature_size = get_features_extractor(
            params.feature_extractor_name
        )

        # self.classifier = nn.Linear(self.feature_size*self.params.nb_samples, 6)
        self.classifier = nn.Linear(self.feature_size, 6)

    def forward(self, x):
        features = self.features_extractor(x)
        output = self.classifier(features)
        return output


class TDCNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # get features extractor
        self.features_extractor, self.feature_size = get_features_extractor(
            params.feature_extractor_name
        )
        
        # self.classifier = nn.Linear(feature_size*self.params.nb_samples, 6)
        # self.classifier = nn.Linear(feature_size, 6)
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.feature_size, self.feature_size // 2, 1),
            nn.GELU(),
            nn.Conv2d(self.feature_size // 2, self.feature_size // 4, 1),
            nn.GELU(),
            nn.Conv2d(self.feature_size // 4, self.feature_size // 8, 1),
            nn.GELU(),
            nn.Conv2d(self.feature_size // 8, self.feature_size // 16, 1),
            nn.GELU(),
            nn.Conv2d(self.feature_size // 16, self.feature_size // 32, 1),
        )

        self.feature_size_mlp = (
            self.conv_block(torch.randn(1, self.feature_size, 4, 4)).shape[1] * 4 * 4
        )

        self.mlp = nn.Sequential(nn.Linear(self.feature_size_mlp, 6))

    def forward(self, x):
        features = []
        for tiles in x:
            feature = self.features_extractor(tiles)
            features.append(feature)
        features = torch.stack(features)
        # rearranged_features = rearrange(features, "p c d -> (p1 p2) c d", p1=4)
        rearranged_features = rearrange(features, "b (p1 p2) d -> b d p1 p2", p1=4)
        features = self.conv_block(rearranged_features)
        output = self.mlp(rearrange(features, "b c h w -> b (c h w)"))
        # output = self.classifier(rearrange(features, "p c d -> (p c d)"))
        return output
