import os
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional

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
    wandb_project: str = f"test-dlmi"
    root_dir: str = os.getcwd()  # root_dir

    # basic params
    seed_everything: Optional[int] = None  # seed for the whole run
    gpu: int = 1  # number or gpu
    max_epochs: int = 30  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")

    # modes
    tune_lr: bool = False  # tune the model on first run
    dev_run: bool = False
    train: bool = True

    # best model for prediction
    best_model: str = ""


@dataclass
class NetworkParams:
    feature_extractor_name: str = "resnet18"
    network_name: str = "Baseline"
    weight_checkpoints: str = ""
    artifact: str = ""

    nb_sample: int = 25

    # MLP parameters
    dropout: float = 0.75
    normalization: str = "BatchNorm1d"
    activation: str = "GELU"


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer: str = "Adam"  # Optimizer default vit: AdamW, default resnet50: Adam
    lr: float = 0.003  # learning rate,               default = 5e-4
    min_lr: float = 5e-9  # min lr reached at the end of the cosine schedule
    weight_decay: float = 1e-8

    # Scheduler
    scheduler: bool = True
    warmup_epochs: int = 5
    max_epochs: int = 20


@dataclass
class DatasetParams:
    """Dataset Parameters"""

    dataset_name: Optional[str] = "BaseDataset"  # dataset, use <Dataset>Eval for FT
    root_dataset: Optional[str] = osp.join(os.getcwd(), "assets", "mvadlmi")

    # dataset
    split_val: float = 0.2
    patch_size: int = 32
    percentage_blank: float = 0.2
    nb_samples: int = 16

    # dataloader
    num_workers: int = 1  # number of workers for dataloadersint
    batch_size: int = 2  # batch_size


@dataclass
class Parameters:
    """base options."""

    hparams: Hparams = Hparams()
    data_param: DatasetParams = DatasetParams()
    network_param: NetworkParams = NetworkParams()
    optim_param: OptimizerParams = OptimizerParams()

    network_param.nb_samples = data_param.nb_samples

    def __post_init__(self):
        """Post-initialization code"""
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)

        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
