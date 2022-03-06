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

################################## Global parameters ##################################


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb
    wandb_entity: str = "attributes_classification_celeba"  # name of the project
    debug: bool = False
    test: bool = True
    wandb_project: str = f"{'test-'*test}dlmi"
    root_dir: str = os.getcwd()  # root_dir

    # basic params
    seed_everything: Optional[int] = None  # seed for the whole run
    gpu: int = 0  # number or gpu
    max_epochs: int = 30  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")
    enable_progress_bar: bool = True

    # modes
    tune_lr: bool = False  # tune the model on first run
    tune_batch_size: bool = True
    dev_run: bool = False
    train: bool = True

    # for inference and test
    best_model: str = ""


@dataclass
class NetworkParams:
    feature_extractor_name: str = "resnet34"
    network_name: str = "Segmentation"
    weight_checkpoints: str = ""
    artifact: str = ""

    # MLP parameters
    dropout: float = 0.1
    normalization: str = "BatchNorm1d"
    activation: str = "GELU"


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer: str = "Adam"  # Optimizer default vit: AdamW, default resnet50: Adam
    lr: float = 5e-4  # learning rate,               default = 5e-4
    min_lr: float = 5e-9  # min lr reached at the end of the cosine schedule
    weight_decay: float = 0.0

    # Scheduler
    scheduler: bool = True
    warmup_epochs: int = 5
    max_epochs: int = 30


@dataclass
class DatasetParams:
    """Dataset Parameters"""

    dataset_name: str = "PatchSegDataset"  # dataset, use <Dataset>Eval for FT
    root_dataset: str = osp.join(os.getcwd(), "assets", "mvadlmi")

    # dataset
    split_val: float = 0.1
    patch_size: int = 512
    percentage_blank: float = 0.5
    nb_samples: int = 4

    # dataloader
    num_workers: int = 4  # number of workers for dataloaders
    batch_size: int = 4  # batch_size

    # for segmentation
    data_provider: str = "karolinska"
    # merge_cls: bool = True # Only for radboud
    image_size: int = 512


@dataclass
class MetricParams:

    # list_metrics: List[str] = list_field(
    #     "Accuracy", "AUROC", "F1", "Recall", "Precision"
    # )
    list_metrics: List[str] = list_field("Accuracy", "Recall", "Precision", "F1", "IoU")
    # list_metrics: List[str] = list_field("IoU")
    num_classes: int = 3
    pixel_wise_parameters: Dict[str, Any] = dict_field(
        dict(average="weighted", mdmc_average="global")
    )
    name_module: str = "MetricsModuleSegmentation"


@dataclass
class CallbacksParams:
    log_freq_img: int = 5
    log_nb_img: int = 8
    log_nb_patches: int = 6


@dataclass
class Parameters:
    """base options."""

    hparams: Hparams = Hparams()
    data_param: DatasetParams = DatasetParams()
    network_param: NetworkParams = NetworkParams()
    optim_param: OptimizerParams = OptimizerParams()
    metric_param: MetricParams = MetricParams()
    callbacks_param: CallbacksParams = CallbacksParams()

    def __post_init__(self):
        """Post-initialization code"""
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)

        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

        self.network_param.nb_samples = self.data_param.nb_samples
        self.network_param.data_provider = self.data_param.data_provider
        self.data_param.feature_extractor_name = (
            self.network_param.feature_extractor_name
        )

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
