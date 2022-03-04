import timm
import torch
import torch.nn as nn

from einops import rearrange

from models.MLP import MLP


class Baseline(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )
        self.features_extractor.reset_classifier(0)
        in_shape = self.features_extractor(torch.randn(1, 3, 224, 224)).shape[1]

        self.mlp = MLP(in_shape * params.nb_samples, params)

    def forward(self, x):
        features = []
        for batch in x:
            # features.append(self.features_extractor(batch.permute(0, 3, 1, 2)))
            # TODO fix the permute issue
            features.append(self.features_extractor(batch))
            # features.append(
            #     self.features_extractor.forward_features(
            #         batch.permute(0, 3, 1, 2)
            #     ).squeeze()
            # )
        # bs, n_patches, h, w, c
        features = torch.stack(features)

        output = self.mlp(rearrange(features, "b p d -> b (p d)"))
        return output
