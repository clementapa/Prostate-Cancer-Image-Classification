import importlib
import os
import wandb

from config.hparams import Parameters
from datasets.datamodule import BaseDataModule


def get_net(network_name, network_param):
    """
    Get Network Architecture based on arguments provided
    """
    mod = importlib.import_module(f"models.{network_name}")
    net = getattr(mod, network_name)
    return net(network_param)


def get_dataset(dataset_name, dataset_param):
    """
    Get Network Architecture based on arguments provided
    """
    mod = importlib.import_module(f"datasets.{dataset_name}")
    dataset = getattr(mod, dataset_name)
    return dataset(dataset_param)


def get_datamodule(data_param):
    """
    Fetch Datamodule Function Pointer
    """
    return BaseDataModule(data_param)


def get_artifact(name: str, type: str) -> str:
    """Artifact utilities
    Extracts the artifact from the name by downloading it locally>
    Return : str = path to the artifact
    """
    if name != "" and name is not None:
        artifact = wandb.run.use_artifact(name, type=type)
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        return file_path
    else:
        return None


def parse_params(parameters: Parameters) -> dict:
    wdb_config = {}
    for k, v in vars(parameters).items():
        for key, value in vars(v).items():
            wdb_config[f"{k}-{key}"] = value
    return wdb_config


def import_class(name, instantiate=None):

    namesplit = name.split(".")
    module = importlib.import_module(".".join(namesplit[:-1]))
    imported_class = getattr(module, namesplit[-1])

    if imported_class:
        if instantiate is not None:
            return imported_class(**instantiate)
        else:
            return imported_class
    raise Exception("Class {} can be imported".format(import_class))
