from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from math import sqrt
from einops import rearrange

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import _METRIC, _PATH, STEP_OUTPUT
from torchvision.utils import make_grid

from utils.constant import MEAN, STD, DICT_COLORS
import utils.metrics as metrics


class AutoSaveModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        config,
        project,
        entity,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        every_n_val_epochs: Optional[int] = None,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            every_n_val_epochs,
        )
        self.config = config
        self.project = project
        self.entity = entity

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, _METRIC],
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"), device=current.device
            )

        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(
                self.best_k_models, key=self.best_k_models.get
            )
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates.get("epoch")
            step = monitor_candidates.get("step")
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor} reached {current:0.5f}"
                f' (best {self.best_model_score:0.5f}), saving model to "{filepath}" as top {k}'
            )
        trainer.save_checkpoint(filepath, self.save_weights_only)

        if del_filepath is not None and filepath != del_filepath:
            trainer.training_type_plugin.remove_checkpoint(del_filepath)

        reverse = False if self.mode == "min" else True
        score = sorted(self.best_k_models.values(), reverse=reverse)
        indices = [(i + 1) for i, x in enumerate(score) if x == current]
        alias = f"top-{indices[0]}"  #
        name = f"{wandb.run.name}"  # name of the model
        model_artifact = wandb.Artifact(type="model", name=name, metadata=self.config)
        model_artifact.add_file(filepath)
        wandb.log_artifact(model_artifact, aliases=[alias])

        # ------------- Clean up previous version -----------------

        if self.verbose:  # only log when there are already 5 models
            epoch = monitor_candidates.get("epoch")
            step = monitor_candidates.get("step")
            rank_zero_info(f"Saved '{name}' weights to wandb")

    def del_artifacts(self):
        api = wandb.Api(overrides={"project": self.project, "entity": self.entity})
        artifact_type, artifact_name = "model", f"{wandb.run.name}"
        try:
            for version in api.artifact_versions(artifact_type, artifact_name):
                # Clean previous versions with the same alias, to keep only the latest top k.
                if (
                    len(version.aliases) == 0
                ):  # this means that it does not have the latest alias
                    # either this works, or I will have to remove the model with the alias first then log the next
                    version.delete()
        except:
            print("error in del artifact to ignore")
            return

    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        self.del_artifacts()
        return super().on_exception(trainer, pl_module, exception)

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.del_artifacts()


class LogMetricsCallback(Callback):
    def __init__(self, params):
        self.params = params

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        device = pl_module.device

        self.metrics_module_train = getattr(metrics, self.params.name_module)(
            "train", self.params, device
        )

        self.metrics_module_validation = getattr(metrics, self.params.name_module)(
            "val", self.params, device
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the train batch ends."""

        _, y = batch
        self.metrics_module_train.update_metrics(outputs["softmax"], y)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_module_train.log_metrics("train/", pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        _, y = batch
        self.metrics_module_validation.update_metrics(outputs["softmax"], y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_validation.log_metrics("val/", pl_module)


class BaseLogImages(Callback):
    def __init__(self, log_freq_img, log_nb_img, log_nb_patches) -> None:
        super().__init__()
        self.log_freq_img = log_freq_img
        self.log_nb_img = log_nb_img
        self.log_nb_patches = log_nb_patches

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq_img == 0:
            self.log_images("val", batch, self.log_nb_img, self.log_nb_patches, outputs)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq_img == 0:
            self.log_images(
                "train", batch, self.log_nb_img, self.log_nb_patches, outputs
            )

    def log_images(self, name, batch, n, p, outputs):
        raise NotImplementedError(f"Should be implemented in derived class!")


class LogImagesClassification(BaseLogImages):
    def __init__(self, log_freq_img, log_nb_img, log_nb_patches) -> None:
        super().__init__(log_freq_img, log_nb_img, log_nb_patches)

    def log_images(self, name, batch, n, p, outputs):

        x, y = batch
        images = x[:n, :p].detach().cpu()
        labels = np.array(y[:n].cpu())
        preds = np.array(outputs["softmax"][:n].argmax(dim=1).cpu())

        samples = []
        for i in range(len(images)):

            bg_image = make_grid(images[i]).numpy().transpose((1, 2, 0))
            bg_image = STD * bg_image + MEAN
            bg_image = np.clip(bg_image, 0, 1)

            samples.append(
                wandb.Image(
                    bg_image, caption=f"label: {labels[i]}, prediction: {preds[i]}"
                )
            )

        wandb.log({f"{name}/predictions": samples})


class LogImagesSegmentation(BaseLogImages):
    def __init__(self, log_freq_img, log_nb_img, log_nb_patches, data_provider) -> None:
        super().__init__(log_freq_img, log_nb_img, log_nb_patches)

        self.data_provider = data_provider

    def log_images(self, name, batch, n, p, outputs):

        x, y = batch

        images = x[:n].detach().cpu()
        labels = y[:n].cpu()
        preds = outputs["softmax"][:n].argmax(dim=1).cpu()

        samples = []

        for i in range(len(images)):

            bg_image = images[i].numpy().transpose((1, 2, 0))
            bg_image = STD * bg_image + MEAN
            bg_image = np.clip(bg_image, 0, 1)

            prediction = np.array(preds[i])
            true_mask = np.array(labels[i])

            samples.append(
                wandb.Image(
                    bg_image,
                    masks={
                        "prediction": {
                            "mask_data": prediction,
                            "class_labels": DICT_COLORS[self.data_provider],
                        },
                        "ground_truth": {
                            "mask_data": true_mask,
                            "class_labels": DICT_COLORS[self.data_provider],
                        },
                    },
                )
            )

        wandb.log({f"{name}/predictions": samples})


class LogImagesSegmentationClassification(Callback):
    def __init__(self, log_freq_img, log_nb_img, log_nb_patches, data_provider) -> None:
        super().__init__()
        self.log_freq_img = log_freq_img
        self.log_nb_img = log_nb_img
        self.log_nb_patches = log_nb_patches

        self.data_provider = data_provider

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq_img == 0:
            self.log_images(
                "val",
                batch,
                self.log_nb_img,
                self.log_nb_patches,
                outputs,
                pl_module.model.seg_model,
                pl_module.network_param,
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq_img == 0:
            self.log_images(
                "train",
                batch,
                self.log_nb_img,
                self.log_nb_patches,
                outputs,
                pl_module.model.seg_model,
                pl_module.network_param,
            )

    def log_images(self, name, batch, n, p, outputs, seg_model, params):

        x, y = batch

        if len(x.shape) != 5:
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) c p1 p2",
                h=int(sqrt(params.nb_samples)),
                p1=params.patch_size,
                p2=params.patch_size,
            )

        images = x[:n, :p].detach()
        labels = y[:n].cpu()

        preds = outputs["softmax"][:n].argmax(dim=1).cpu()

        batch_masks = []
        for b in images:
            masks = seg_model(b).argmax(dim=1).cpu()
            batch_masks.append(masks)
        batch_masks = torch.stack(batch_masks)

        images = images.cpu()

        samples = []
        for i in range(len(images)):

            bg_image = make_grid(images[i]).numpy().transpose((1, 2, 0))
            bg_image = STD * bg_image + MEAN
            bg_image = np.clip(bg_image, 0, 1)

            prediction = make_grid(batch_masks[i].unsqueeze(1)).numpy()[0]

            samples.append(
                wandb.Image(
                    bg_image,
                    masks={
                        "prediction": {
                            "mask_data": prediction,
                            "class_labels": DICT_COLORS[self.data_provider],
                        },
                    },
                    caption=f"label: {labels[i]}, prediction: {preds[i]}",
                )
            )

        wandb.log({f"{name}/predictions_seg": samples})
