import timm
import torch.nn as nn 
import torch
from einops import rearrange

class SimpleModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )
        self.features_extractor.reset_classifier(0)
        in_shape = self.features_extractor(torch.randn(1, 3, 384, 384)).shape[1]

        # self.classifier = nn.Linear(in_shape*self.params.nb_samples, 6)
        self.classifier = nn.Linear(in_shape, 6)

    def forward(self, x):
        features = self.features_extractor(x)
        output = self.classifier(features)
        return output