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
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.features_extractor = timm.create_model(
            self.params.feature_extractor_name, pretrained=True
        )
        self.features_extractor.reset_classifier(0)
        in_shape = self.features_extractor(torch.randn(1, 3, 224, 224)).shape[1]

        self.feature_selector = nn.Sequential(
            nn.Linear(in_shape+3, 1),
            nn.Sigmoid()
        )

        # self.mlp = MLP(params.bottleneck_shape * params.nb_samples, params)
        self.mlp = MLP(in_shape+3, params)

        # load seg_model
        name_artifact = f"attributes_classification_celeba/test-dlmi/{params.wb_run_seg}:top-1"
        path_to_model = get_artifact(f"{name_artifact}", "model")
        # path_to_model = "/home/younesbelkada/Travail/MVA/DeepMedical/Prostate-Cancer-Image-Classification/artifacts/expert-surf-171:v7/epoch-19-val_loss=0.17.ckpt"

        base_module = BaseModuleForInference(params)
        base_module.load_state_dict(torch.load(path_to_model)['state_dict'])
        self.seg_model = base_module.model
        # self.seg_model._requires_grad(False)

    def forward(self, x):
        features = []
        scores = []
        with torch.no_grad():
            for batch in x:
                # TODO fix the permute issue
                feature = self.features_extractor(batch)
                seg_mask = self.seg_model(batch).argmax(dim=1)

                score = seg_max_to_score(seg_mask)

                transformed_feature = self.feature_selector(torch.cat([feature, score], dim=-1))
                features.append(transformed_feature)
                scores.append(score)
            most_relevant_patch = torch.argmax(torch.stack(features), dim=1)
        # bs, n_patches, h, w, c
        # features = torch.stack(features)
        scores = torch.stack(scores)
        important_patches = x[torch.arange(x.size(0)), most_relevant_patch.squeeze()]
        important_scores = scores[torch.arange(scores.size(0)), most_relevant_patch.squeeze()]

        feature = self.features_extractor(important_patches)

        probas = self.feature_selector(torch.cat([feature, important_scores], dim=-1))

        # output = self.mlp(rearrange(features, "b p d -> b (p d)"))
        output = self.mlp(torch.cat([feature, important_scores], dim=-1))
        return output, probas