import torch
import torch.nn as nn
from einops import rearrange
from utils.agent_utils import get_seg_model
from utils.dataset_utils import seg_max_to_score

from models.BaseModels import BaseTopPatchMethods


class TopPatchWithSeg(BaseTopPatchMethods):
    def __init__(self, params):
        super().__init__(self, params)

        self.patch_selector = nn.Sequential(
            nn.Linear(self.in_shape + 3, 1), nn.Sigmoid()
        )

        # load seg_model
        self.seg_model = get_seg_model(params)

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


class TopKPatchWithSeg(BaseTopPatchMethods):
    def __init__(self, params):
        super().__init__(self, params)

        self.patch_selector = nn.Sequential(
            nn.Linear(self.in_shape + 3, 1), nn.Sigmoid()
        )

        # load seg_model
        self.seg_model = get_seg_model(params)

    def forward(self, x):
        top_k_patches = []
        scores = []
        with torch.no_grad():
            for batch in x:
                seg_mask = self.seg_model(batch).argmax(dim=1)
                score = seg_max_to_score(seg_mask, seg_mask.shape[-1])
                top_k_patches.append(
                    torch.topk(score[:, -1], x.shape[1] // 2, dim=-1).indices
                )
                scores.append(score)
            # most_relevant_patch = torch.argmax(torch.stack(scores), dim=1)
        # bs, n_patches, h, w, c
        # features = torch.stack(features)
        top_k_patches = torch.stack(top_k_patches)
        filtered_features = []

        for i, top_k in enumerate(top_k_patches):
            selected_patches = x[i, top_k]
            features = self.features_extractor(selected_patches)
            filtered_features.append(features)
        mean_filtered_features = torch.stack(filtered_features).mean(dim=1)
        output = self.classifier(mean_filtered_features)
        return output


class OnlySeg(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.classifier = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 6))

        # load seg_model
        self.seg_model = get_seg_model(params)

    def forward(self, x):
        scores = []
        with torch.no_grad():
            for batch in x:
                seg_mask = self.seg_model(batch).argmax(dim=1)
                score = seg_max_to_score(seg_mask, seg_mask.shape[-1])
                scores.append(score)
        # bs, n_patches, h, w, c
        # features = torch.stack(features)
        scores = torch.stack(scores).mean(dim=1)
        output = self.classifier(scores)
        return output


class OnlySeg2(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params

        in_shape = params.nb_samples * 3

        self.norm = getattr(nn, params.normalization)
        self.activation = getattr(nn, params.activation)

        self.classifier = nn.Sequential(
            nn.Linear(in_shape, in_shape * 2),
            self.norm(2 * in_shape),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(in_shape * 2, in_shape * 4),
            self.norm(4 * in_shape),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(in_shape * 4, in_shape * 8),
            self.norm(8 * in_shape),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(in_shape * 8, in_shape * 16),
            self.norm(16 * in_shape),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(in_shape * 16, in_shape * 8),
            self.norm(8 * in_shape),
            self.activation(),
            nn.Dropout(params.dropout),
            nn.Linear(in_shape * 8, 6),
        )

    def forward(self, x):
        scores = []

        for batch_patch in x:

            with torch.no_grad():
                seg_mask = self.seg_model(batch_patch).argmax(dim=1)

            seg_score = seg_max_to_score(seg_mask, seg_mask.shape[-1])

            scores.append(seg_score)

        scores = torch.stack(scores)
        scores = rearrange(scores, "b p d -> b (p d)")

        output = self.classifier(scores)

        return output
