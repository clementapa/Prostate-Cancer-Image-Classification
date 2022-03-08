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

        self.feature_selector = nn.Sequential(
            nn.Linear(in_shape, 1),
            nn.Sigmoid()
        )

        # self.mlp = MLP(params.bottleneck_shape * params.nb_samples, params)
        self.mlp = MLP(in_shape, params)

    def forward(self, x):
        features = []
        with torch.no_grad():
            for batch in x:
                # TODO fix the permute issue
                feature = self.features_extractor(batch)
                transformed_feature = self.feature_selector(feature)
                features.append(transformed_feature)
            most_relevant_patch = torch.argmax(torch.stack(features), dim=1)
        # bs, n_patches, h, w, c
        # features = torch.stack(features)
        important_patches = x[torch.arange(x.size(0)), most_relevant_patch.squeeze()]
        feature = self.features_extractor(important_patches)

        # output = self.mlp(rearrange(features, "b p d -> b (p d)"))
        output = self.mlp(feature)
        return output
