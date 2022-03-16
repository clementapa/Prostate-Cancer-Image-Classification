import torch.nn as nn

from models.classifier.MLP import MLP
from utils.agent_utils import get_features_extractor


class BaseTopPatchMethods(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # get features extractor
        self.features_extractor, self.feature_size = get_features_extractor(
            params.feature_extractor_name
        )

        self.norm = getattr(nn, params.normalization)
        self.activation = getattr(nn, params.activation)

        # get classifier
        if self.params.classifier_name == "MLP":
            self.classifier = MLP(self.feature_size + 1, params)
        elif self.params.classifier_name == "Linear":
            self.classifier = nn.Linear(self.feature_size + 1, 6)
        elif self.params.classifier_name == "Multiple Linear":
            self.classifier = nn.Sequential(
                # nn.Linear(feature_size+4, feature_size+4),
                # self.norm(feature_size+4),
                # self.activation(),
                # nn.Dropout(params.dropout),
                nn.Linear(self.feature_size + 1, (self.feature_size + 1) // 2),
                # self.norm((feature_size+4)//2),
                self.activation(),
                nn.Dropout(params.dropout),
                nn.Linear((self.feature_size + 1) // 2, 6),
            )
        else:
            raise NotImplementedError(
                "Classifier not implemented ! MLP, Linear or Multiple Linear"
            )

    def forward(self, x):
        raise NotImplementedError("To implement in the derived classes !")
