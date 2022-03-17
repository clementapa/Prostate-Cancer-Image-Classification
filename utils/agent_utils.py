import errno
import importlib
import os
import timm

import wandb
from config.hparams import Parameters
from datasets.datamodule import BaseDataModule

import torch


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


def get_datamodule(mode, data_param):
    """
    Fetch Datamodule Function Pointer
    """
    return BaseDataModule(mode, data_param)


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


def get_seg_model(params):
    # load seg_model
    name_artifact = (
        f"attributes_classification_celeba/test-dlmi/{params.wb_run_seg}:top-1"
    )
    artifact = wandb.use_artifact(name_artifact)

    model = get_net("Segmentation", params.seg_param)
    path_to_model = artifact.download()
    pth = torch.load(
        os.path.join(path_to_model, os.listdir(path_to_model)[0]),
        map_location=torch.device("cpu"),
    )["state_dict"]
    pth = {".".join(k.split(".")[1:]): v for k, v in pth.items()}
    model.load_state_dict(pth)

    return model


def get_classif_model(params):
    # load seg_model
    name_artifact = (
        f"attributes_classification_celeba/test-dlmi/{params.wb_run_classif}:top-1"
    )
    artifact = wandb.use_artifact(name_artifact)

    # model = get_net("Classification", params.classif_param.network_name)
    mod = importlib.import_module(f"models.Classification")
    model = getattr(mod, params.classif_param.network_name)(params.classif_param)
    path_to_model = artifact.download()
    pth = torch.load(
        os.path.join(path_to_model, os.listdir(path_to_model)[0]),
        map_location=torch.device("cpu"),
    )["state_dict"]
    pth = {".".join(k.split(".")[1:]): v for k, v in pth.items()}
    model.load_state_dict(pth)

    return model


def get_features_extractor(feature_extractor_name):
    features_extractor = timm.create_model(feature_extractor_name, pretrained=True)
    features_extractor.reset_classifier(0)
    features_size = features_extractor(torch.randn(1, 3, 224, 224)).shape[1]

    return features_extractor, features_size


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


def create_dir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
