from utils.agent_utils import import_class
import torch

"""
https://torchmetrics.readthedocs.io/en/stable/references/modules.html#base-class MODULE METRICS
"""


class BaseMetricsModule:
    def __init__(self, set_name, params, device) -> None:
        self.device = device

    def update_metrics(self, x, y):

        for _, m in self.dict_metrics.items():
            # metric on current batch
            m(x, y)  # update metrics (torchmetrics method)

    def log_metrics(self, name, pl_module):

        for k, m in self.dict_metrics.items():

            # metric on all batches using custom accumulation
            metric = m.compute()
            if metric.shape != torch.Size([]):
                for i, v in enumerate(metric):
                    pl_module.log(f"{name+k}_{i}", v)
            else:
                pl_module.log(name + k, metric)

            # Reseting internal state such that metric ready for new data
            m.reset()
            m.to(self.device)


class MetricsModuleClassification(BaseMetricsModule):
    def __init__(self, set_name, params, device) -> None:
        super().__init__(set_name, params, device)
        """
        metrics : list of name metrics e.g ["Accuracy", "IoU"]
        set_name: val/train/test
        """
        self.device = device
        dict_metrics = {}
        for name in params.list_metrics:
            instance = import_class("torchmetrics." + name)(
                compute_on_step=False,
                num_classes=params.num_classes,
                average=params.average,
            )
            dict_metrics[name.lower()] = instance.to(device)

        dict_metrics["auroc_class"] = import_class("torchmetrics.AUROC")(
            compute_on_step=False,
            num_classes=params.num_classes,
            average=None,
        )

        self.dict_metrics = dict_metrics


class MetricsModuleSegmentation(BaseMetricsModule):
    def __init__(self, set_name, params, device) -> None:
        super().__init__(set_name, params, device)
        """
        metrics : list of name metrics e.g ["Accuracy", "IoU"]
        set_name: val/train/test
        """

        dict_metrics = {}
        if set_name != "train":
            for name in params.list_metrics:
                if name != "IoU":
                    instance = import_class("torchmetrics." + name)(
                        compute_on_step=False,
                        num_classes=params.num_classes,
                        **params.pixel_wise_parameters,
                    )
                else:
                    instance = import_class("torchmetrics." + name)(
                        compute_on_step=False,
                        num_classes=params.num_classes,
                    )
                dict_metrics[name.lower()] = instance.to(device)
        else:
            dict_metrics["iou"] = import_class("torchmetrics.IoU")(
                compute_on_step=False, num_classes=params.num_classes
            ).to(device)

        self.dict_metrics = dict_metrics
