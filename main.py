import faulthandler

faulthandler.enable()

from pytorch_lightning.loggers import WandbLogger

# Standard libraries
import wandb
from agents.BaseTrainer import BaseTrainer
from config.hparams import Parameters
from utils.agent_utils import parse_params


def main():
    parameters = Parameters.parse()

    # initialize wandb instance
    wdb_config = parse_params(parameters)

    tags = [
        parameters.hparams.MODE,
        parameters.network_param.network_name,
        parameters.data_param.dataset_name,
        f"patch_size: {parameters.data_param.patch_size}",
        f"Backbone: {parameters.network_param.feature_extractor_name}",
    ]
    if parameters.hparams.MODE == "Segmentation":
        tags += [f"provider: {parameters.network_param.data_provider}"]
    elif parameters.hparams.MODE == "Classification":
        tags += [f"nb_sample: {parameters.data_param.nb_samples}"]
    else:
        tags += [f"nb_sample: {parameters.data_param.nb_samples}"]

    if parameters.hparams.train:
        wandb_run = wandb.init(
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            job_type="train",
            tags=tags,
        )

        wandb_logger = WandbLogger(
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
        )

        agent = BaseTrainer(parameters, wandb_logger)
        agent.run()
    else:
        wandb_run = wandb.init(
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            job_type="test",
        )

        wandb_logger = WandbLogger(
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
        )
        agent = BaseTrainer(parameters, wandb_logger, wb_run=wandb_run)
        agent.predict()


if __name__ == "__main__":
    main()
