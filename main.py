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

    if parameters.hparams.train:
        wandb_run = wandb.init(
            # vars(parameters),  # FIXME use the full parameters
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            job_type="train",
            tags=[
                parameters.network_param.network_name,
                parameters.data_param.dataset_name,
                f"patch_size:{parameters.data_param.patch_size}",
                parameters.optim_param.optimizer,
                parameters.data_param.data_provider,
                f"nb_sample:{parameters.data_param.nb_samples}",
            ],
        )

        wandb_logger = WandbLogger(
            config=wdb_config,  # vars(parameters),  # FIXME use the full parameters
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            # save_dir=parameters.hparams.save_dir,
        )

        agent = BaseTrainer(parameters, wandb_logger, wandb_run)
        agent.run()
    else:
        wandb_run = wandb.init(
            # vars(parameters),  # FIXME use the full parameters
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            job_type="test",
        )

        wandb_logger = WandbLogger(
            config=wdb_config,  # vars(parameters),  # FIXME use the full parameters
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            # save_dir=parameters.hparams.save_dir,
        )
        agent = BaseTrainer(parameters, wandb_logger, wb_run=wandb_run)
        agent.predict()


if __name__ == "__main__":
    main()
