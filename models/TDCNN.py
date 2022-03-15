import timm
import torch.nn as nn 
import torch
from einops import rearrange

class TDCNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )
        self.features_extractor.reset_classifier(0)
        in_shape = self.features_extractor(torch.randn(1, 3, 224, 224)).shape[1]

        # self.classifier = nn.Linear(in_shape*self.params.nb_samples, 6)
        # self.classifier = nn.Linear(in_shape, 6)
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_shape, in_shape//2, 1 
            ),
            nn.GELU(),
            nn.Conv2d(
                in_shape//2, in_shape//4, 1 
            ),
            nn.GELU(),
            nn.Conv2d(
                in_shape//4, in_shape//8, 1
            ),
            nn.GELU(),
            nn.Conv2d(
                in_shape//8, in_shape//16, 1
            ),
            nn.GELU(),
            nn.Conv2d(
                in_shape//16, in_shape//32, 1
            ),
        )

        in_shape_mlp = self.conv_block(torch.randn(1, in_shape, 4, 4)).shape[1] * 4 * 4 

        self.mlp = nn.Sequential(
            nn.Linear(in_shape_mlp, 6)
        )

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