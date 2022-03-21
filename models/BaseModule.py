import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils.agent_utils import import_class, get_seg_model

from models.Segmentation import Segmentation
import models.Classification as Classification
import models.ClassifWithSeg as ClassifWithSeg


class BaseModule(LightningModule):
    def __init__(self, mode, network_param, optim_param, loss_param):
        """method used to define our model parameters"""
        super(BaseModule, self).__init__()

        self.network_param = network_param

        # loss function
        self.loss = import_class(loss_param.name, instantiate=loss_param.params)

        # optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        # model
        if mode == "Segmentation":
            self.model = Segmentation(network_param)
        elif mode == "Classification":
            self.model = getattr(Classification, network_param.network_name)(
                network_param
            )
        else:
            self.model = getattr(ClassifWithSeg, network_param.network_name)(
                network_param
            )
            

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss, softmax = self._get_preds_loss_accuracy(batch)

        # Log loss
        self.log("train/loss", loss)

        return {"loss": loss, "softmax": softmax.detach()}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, softmax = self._get_preds_loss_accuracy(batch)

        # Log loss
        self.log("val/loss", loss)

        return {"softmax": softmax}

    def predict_step(self, batch, batch_idx):

        x, _ = batch
        output = self(x)

        if isinstance(output, tuple):
            output = output[0].argmax(dim=-1)
        else:
            output = output.argmax(dim=-1)

        return output

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(
            self.parameters(), lr=self.lr, weight_decay=self.optim_param.weight_decay
        )

        if self.optim_param.scheduler:
            scheduler = {
                "scheduler": getattr(
                    torch.optim.lr_scheduler, self.optim_param.scheduler_name
                )(optimizer, **self.optim_param.scheduler_params),
                "monitor": "val/loss",
            }

            return [[optimizer], [scheduler]]

        return optimizer

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        output = self(x)

        loss = self.loss(output, y)

        if isinstance(output, tuple):
            softmax = F.softmax(output[0], dim=0)
        else:
            softmax = F.softmax(output, dim=0)

        return loss, softmax
