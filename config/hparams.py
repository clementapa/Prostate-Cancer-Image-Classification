import os
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional
from simple_parsing.helpers import Serializable, choice, dict_field, list_field

import pytorch_lightning as pl
import simple_parsing
import torch
import torch.optim

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
    max_epochs: int = 60  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")
    enable_progress_bar: bool = True

    # modes
    tune_lr: bool = False
    tune_batch_size: bool = False
    dev_run: bool = False
    train: bool = True

    # for inference and test
    best_model: str = "skilled-gorge-229"

    # Segmentation, Classification & Classif_WITH_Seg
    MODE: str = "Segmentation"


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

    dataset_name: str = "PatchSegDataset"  # dataset, use <Dataset>Eval for FT
    root_dataset: str = osp.join(os.getcwd(), "assets", "mvadlmi")
    path_patches: str = osp.join(os.getcwd(), "assets", "dataset_patches")

    # Static or OFY = On-the-fly TODO
    MODE: str = "Static"  # doesnot implemented yet

    # Patches params
    patch_size: int = 384
    percentage_blank: float = 0.5
    level: int = 1

    # dataset params
    split_val: float = 0.1
    nb_samples: int = 1 # FIXME

    nb_patches: int = 4 # FIXME 
    resized_patch: int = 256

    # dataloader
    num_workers: int = 2  # number of workers for dataloaders
    batch_size: int = 4  # batch_size

    train_artifact: str = "attributes_classification_celeba/dlmi/train_256_1_0.5:v0"
    # train_artifact: str = "attributes_classification_celeba/dlmi/train_192_1_0.5:v0"
    test_artifact: str = "attributes_classification_celeba/dlmi/test_256_1_0.5:v0"


@dataclass
class CallbacksParams:

    # params log predictions
    log_freq_img: int = 1
    log_nb_img: int = 4
    log_nb_patches: int = 18

    # Early Stopping
    early_stopping: bool = True
    early_stopping_params: Dict[str, Any] = dict_field(
        dict(monitor="val/loss", patience=50, mode="min", verbose=True)
    )


##################################### Classification #############################################


@dataclass
class NetworkClassificationParams:
    feature_extractor_name: str = "resnet34"
    network_name: str = "SimpleModel"
    classifier_name: str = "Multiple Linear"

    # MLP parameters
    dropout: float = 0.0
    normalization: str = "BatchNorm1d"
    activation: str = "ReLU"


@dataclass
class NetworkClassif_WITH_SegParams:
    feature_extractor_name: str = "resnet34"
    network_name: str = "OnlySeg"
    
    classifier_name: str = "Multiple Linear"
    # MLP parameters
    dropout: float = 0.0
    normalization: str = "BatchNorm1d"
    activation: str = "ReLU"

    # Seg Model param
    wb_run_seg: str = "expert-surf-171"


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
