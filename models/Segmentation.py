import torch.nn as nn
import segmentation_models_pytorch as smp

from utils.constant import CLASSES_PER_PROVIDER


class Segmentation(nn.Module):
    """
    https://smp.readthedocs.io/en/latest/
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.seg_model = getattr(smp, params.network_name)(
            encoder_name=params.feature_extractor_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=params.encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params.num_classes  # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.seg_model(x)
