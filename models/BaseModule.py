import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.agent_utils import get_net
from models.Baseline import Baseline


class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param):
        """method used to define our model parameters"""
        super(BaseModule, self).__init__()

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        # model
        self.model = get_net(network_param.network_name, network_param)
        # if network_param.weight_checkpoint is not None:
        #     self.model.load_state_dict(torch.load(
        #         network_param.weight_checkpoint)["state_dict"])

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss
        self.log("train/loss", loss)

        return {"loss": loss, "logits": logits.detach()}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss
        self.log("val/loss", loss)

        return {"logits": logits}

    def predict_step(self, batch, batch_idx):

        x = batch
        output = self(x)
        output = torch.sigmoid(output)

        return output

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(
            self.parameters(), lr=self.lr, weight_decay=self.optim_param.weight_decay
        )

        if self.optim_param.scheduler:
            # scheduler = LinearWarmupCosineAnnealingLR(
            #     optimizer, warmup_epochs=self.optim_param.warmup_epochs, max_epochs=self.optim_param.max_epochs
            # )
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", patience=5, min_lr=5e-6
                ),
                "monitor": "val/loss",
            }

            return [[optimizer], [scheduler]]

        return optimizer

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        output = self(x)

        loss = self.loss(output, y)
        logits = F.softmax(output, dim=0)

        return loss, logits
