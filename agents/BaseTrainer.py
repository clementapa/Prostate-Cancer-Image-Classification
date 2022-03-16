import pytorch_lightning as pl
import torch
import wandb
import pandas as pd
from models.BaseModule import BaseModule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    StochasticWeightAveraging,
    EarlyStopping,
)
from utils.agent_utils import get_artifact, get_datamodule, create_dir
from utils.callbacks import (
    AutoSaveModelCheckpoint,
    LogImagesSegmentation,
    LogImagesClassification,
    LogImagesSegmentationClassification,
    LogMetricsCallback,
)
from utils.logger import init_logger


class BaseTrainer:
    def __init__(self, config, logger=None) -> None:
        self.config = config.hparams
        self.wb_logger = logger
        self.network_param = config.network_param
        self.metric_param = config.metric_param
        self.callbacks_param = config.callbacks_param
        self.data_param = config.data_param
        self.loss_param = config.loss_param
        self.optim_param = config.optim_param

        self.logger = init_logger("BaseTrainer", "INFO")

        self.logger.info(f"MODE : {self.config.MODE}")

        # self.logger.info("Loading artifact...")
        # self.load_artifact(self.network_param, self.data_param)

        self.logger.info("Loading Data module...")
        self.datamodule = get_datamodule(self.config.MODE, self.data_param)

        self.logger.info("Loading Model module...")
        self.pl_model = BaseModule(
            self.config.MODE, self.network_param, self.optim_param, self.loss_param
        )

        if self.config.MODE == "Segmentation":
            self.wb_logger.watch(self.pl_model.model.seg_model)

    def run(self):
        if self.config.tune_batch_size:
            trainer = pl.Trainer(
                logger=self.wb_logger,
                gpus=self.config.gpu,
                auto_scale_batch_size="power",
                log_every_n_steps=1,
                accelerator="auto",
                default_root_dir=self.wb_logger.save_dir,
                enable_progress_bar=self.config.enable_progress_bar,
            )
            trainer.logger = self.wb_logger
            trainer.tune(self.pl_model, datamodule=self.datamodule)

        if self.config.tune_lr:
            trainer = pl.Trainer(
                logger=self.wb_logger,
                gpus=self.config.gpu,
                auto_lr_find=True,
                log_every_n_steps=1,
                accelerator="auto",
                default_root_dir=self.wb_logger.save_dir,
                enable_progress_bar=self.config.enable_progress_bar,
            )
            trainer.logger = self.wb_logger
            trainer.tune(self.pl_model, datamodule=self.datamodule)

        trainer = pl.Trainer(
            logger=self.wb_logger,  # W&B integration
            callbacks=self.get_callbacks(),
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            log_every_n_steps=1,
            fast_dev_run=self.config.dev_run,
            amp_backend="apex",
            enable_progress_bar=self.config.enable_progress_bar,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
        )
        trainer.logger = self.wb_logger
        trainer.fit(self.pl_model, datamodule=self.datamodule)

    def predict(self):
        # return
        trainer = pl.Trainer(gpus=self.config.gpu)
        best_path = f"attributes_classification_celeba/{self.config.wandb_project}/{self.config.best_model}:top-1"
        best_model = get_artifact(best_path, type="model")

        raw_predictions = trainer.predict(
            self.pl_model, self.datamodule, ckpt_path=best_model
        )
        raw_predictions = torch.cat(raw_predictions, axis=0)
        y_pred = raw_predictions.detach().cpu().numpy()
        ids = self.datamodule.dataset.df["image_id"].values

        output_df = pd.DataFrame({"Id": {}, "Predicted": {}})
        output_df["Id"] = ids
        output_df["Predicted"] = y_pred

        create_dir("submissions")

        output_df.to_csv(
            f"submissions/{self.config.best_model}{'-debug'*self.config.debug}.csv",
            index=False,
        )

    # def load_artifact(self, network_param, data_param):
    #     return
    #     # network_param.weight_checkpoint = get_artifact(
    #     #     network_param.artifact, type="model")

    def get_callbacks(self):
        callbacks = [
            RichProgressBar(),
            LearningRateMonitor(),
            StochasticWeightAveraging(),
            LogMetricsCallback(self.metric_param),
        ]

        # Checkpoint
        if self.config.MODE == "Segmentation":
            monitor = "val/iou"
            mode = "max"
        else:
            monitor = "val/auroc"
            mode = "max"

        self.logger.info(f"Checkpoint: monitor {monitor} {mode}")

        wandb.define_metric(monitor, summary=mode)
        save_top_k = 5
        every_n_epochs = 1
        callbacks += [
            AutoSaveModelCheckpoint(  # ModelCheckpoint
                config=(self.network_param).__dict__,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                monitor=monitor,
                mode=mode,
                filename="epoch-{epoch:02d}-"
                + monitor.replace("/", "_")
                + "={"
                + monitor
                + ":.2f}",
                verbose=True,
                dirpath=self.config.weights_path + f"/{str(wandb.run.name)}",
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
                auto_insert_metric_name=False,
            )
        ]  # our model checkpoint callback

        # Early stopping
        if self.callbacks_param.early_stopping:
            callbacks += [EarlyStopping(**self.callbacks_param.early_stopping_params)]

        # Metrics
        if self.config.MODE == "Segmentation":
            callbacks += [
                LogImagesSegmentation(
                    self.callbacks_param.log_freq_img,
                    self.callbacks_param.log_nb_img,
                    self.callbacks_param.log_nb_patches,
                    self.network_param.data_provider,
                )
            ]
        elif self.config.MODE == "Classification":
            callbacks += [
                LogImagesClassification(
                    self.callbacks_param.log_freq_img,
                    self.callbacks_param.log_nb_img,
                    self.callbacks_param.log_nb_patches,
                )
            ]
        elif self.config.MODE == "Classif_WITH_Seg":
            callbacks += [
                LogImagesSegmentationClassification(
                    self.callbacks_param.log_freq_img,
                    self.callbacks_param.log_nb_img,
                    self.callbacks_param.log_nb_patches,
                    self.network_param.seg_param.data_provider,
                )
            ]

        return callbacks
