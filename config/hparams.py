import os
import random
from dataclasses import dataclass
from math import sqrt
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional

import pytorch_lightning as pl
import simple_parsing
import torch
import torch.optim
from simple_parsing.helpers import Serializable, choice, dict_field, list_field
from utils.constant import CLASSES_PER_PROVIDER

################################## Global parameters ##################################


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb
    wandb_entity: str = "attributes_classification_celeba"  # name of the project
    debug: bool = True
    wandb_project: str = f"{'test-'*debug}dlmi"
    root_dir: str = os.getcwd()  # root_dir

    # basic params
    seed_everything: Optional[int] = None  # seed for the whole run
    gpu: int = 0  # number or gpu
    max_epochs: int = 100  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")
    enable_progress_bar: bool = True

    # modes
    tune_lr: bool = False
    tune_batch_size: bool = False
    dev_run: bool = False
    train: bool = True

    # for inference and test
    best_model: str = "feasible-shape-757"

    # Segmentation, Classification & Classif_WITH_Seg
    MODE: str = "Classification"


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    # Optimizer
    optimizer: str = "Adam"  # Optimizer default vit: AdamW, default resnet50: Adam
    lr: float = 5e-4  # learning rate,               default = 5e-4
    min_lr: float = 5e-9  # min lr reached at the end of the cosine schedule
    weight_decay: float = 0.0

    accumulate_grad_batches: int = 1

    # Scheduler
    scheduler: bool = True
    scheduler_name: str = "ReduceLROnPlateau"
    scheduler_params: Dict[str, Any] = dict_field(
        dict(mode="min", patience=5, min_lr=5e-6)
    )


@dataclass
class DatasetParams:
    """Dataset Parameters"""

    dataset_name: str = "ConcatPatchDataset"  # dataset, use <Dataset>Eval for FT
    root_dataset: str = osp.join(os.getcwd(), "assets", "mvadlmi")
    path_patches: str = osp.join(os.getcwd(), "assets", "dataset_patches")

    # dataset static params
    dataset_static: str = True
    recreate_patches: bool = False

    # Patches params
    patch_size: int = 256
    percentage_blank: float = 0.5  # max percentage
    level: int = 1

    # dataset params
    split_val: float = 0.1
    nb_samples: int = 36

    resized_img: int = 768

    # dataloader
    num_workers: int = 2  # number of workers for dataloaders
    batch_size: int = 4  # batch_size


@dataclass
class CallbacksParams:

    # params log predictions
    log_freq_img: int = 1
    log_nb_img: int = 4
    log_nb_patches: int = 10000

    # Early Stopping
    early_stopping: bool = True
    early_stopping_params: Dict[str, Any] = dict_field(
        dict(monitor="val/loss", patience=50, mode="min", verbose=True)
    )


##################################### Classification #############################################


@dataclass
class NetworkClassificationParams:
    feature_extractor_name: str = "tresnet_xl_448"
    network_name: str = "SimpleModel"


@dataclass
class NetworkClassif_WITH_SegParams:
    feature_extractor_name: str = "tresnet_xl_448"
    network_name: str = "SimpleModelWithSeg"

    # classifier_name: str = "Multiple Linear"
    # # MLP parameters
    # dropout: float = 0.0
    # normalization: str = "BatchNorm1d"
    # activation: str = "ReLU"

    # Seg Model param
    wb_run_seg: str = "drawn-dream-632"


@dataclass
class LossClassificationParams:

    name: str = "torch.nn.CrossEntropyLoss"
    params: Dict[str, Any] = dict_field(
        dict(
            reduction="mean",
        )
    )

    # name: str = "models.losses.customized_ce.C_Crossentropy"
    # params: Dict[str, Any] = dict_field(
    #         dict(
    #             alpha=0.4,
    #         )
    #     )


@dataclass
class MetricClassificationParams:

    name_module: str = "MetricsModuleClassification"
    list_metrics: List[str] = list_field(
        "Accuracy", "Recall", "Precision", "F1", "AUROC"
    )
    average: str = "weighted"
    num_classes: int = 6


##################################### Segmentation #############################################


@dataclass
class NetworkSegmentationParams:

    network_name: str = "DeepLabV3Plus"
    feature_extractor_name: str = "resnet152"
    encoder_weights: str = "imagenet"

    # karolinska, radboud, radboud_merged or all
    data_provider: str = "all"


@dataclass
class LossSegmentationParams:

    name: str = "segmentation_models_pytorch.losses.DiceLoss"
    params: Dict[str, Any] = dict_field(dict(mode="multiclass", from_logits=True))


@dataclass
class MetricSegmentationParams:

    name_module: str = "MetricsModuleSegmentation"
    list_metrics: List[str] = list_field("IoU")


@dataclass
class Parameters:
    """base options."""

    hparams: Hparams = Hparams()
    data_param: DatasetParams = DatasetParams()
    optim_param: OptimizerParams = OptimizerParams()
    callbacks_param: CallbacksParams = CallbacksParams()

    def __post_init__(self):
        """Post-initialization code"""
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)

        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

        self.hparams.wandb_project = f"{'test-'*self.hparams.debug}dlmi"

        if self.hparams.MODE == "Segmentation":
            self.network_param = NetworkSegmentationParams()
            self.metric_param = MetricSegmentationParams()

            self.metric_param.num_classes = CLASSES_PER_PROVIDER[
                self.network_param.data_provider
            ]
            self.network_param.num_classes = self.metric_param.num_classes

            self.loss_param = LossSegmentationParams()

        elif self.hparams.MODE == "Classification":
            self.network_param = NetworkClassificationParams()
            self.metric_param = MetricClassificationParams()
            self.loss_param = LossClassificationParams()

        elif self.hparams.MODE == "Classif_WITH_Seg":
            self.network_param = NetworkClassif_WITH_SegParams()
            self.metric_param = MetricClassificationParams()
            self.loss_param = LossClassificationParams()

            self.network_param.seg_param = NetworkSegmentationParams()
            self.network_param.seg_param.num_classes = CLASSES_PER_PROVIDER[
                self.network_param.seg_param.data_provider
            ]

            if self.data_param.dataset_name == "ConcatPatchDataset":
                assert (
                    str(sqrt(self.data_param.nb_samples))[-1] == "0"
                ), f"{self.data_param.nb_samples} has to be squared root"
                self.network_param.nb_samples = self.data_param.nb_samples
                self.network_param.patch_size = self.data_param.patch_size
                self.network_param.resized_img = self.data_param.resized_img
        else:
            raise NotImplementedError(
                f"Mode {self.hparams.MODE} does not exist only Segmentation, Classification or Classif_WITH_Seg!"
            )

        self.hparams.accumulate_grad_batches = self.optim_param.accumulate_grad_batches

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
