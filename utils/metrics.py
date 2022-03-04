from utils.agent_utils import import_class

"""
https://torchmetrics.readthedocs.io/en/stable/references/modules.html#base-class MODULE METRICS
"""


class MetricsModule:
    def __init__(self, set_name, params, device) -> None:
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

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y):

        for _, m in self.dict_metrics.items():
            # metric on current batch
            m(x, y)  # update metrics (torchmetrics method)

    def log_metrics(self, name, pl_module):

        for k, m in self.dict_metrics.items():

            # metric on all batches using custom accumulation
            metric = m.compute()
            pl_module.log(name + k, metric)

            # Reseting internal state such that metric ready for new data
            m.reset()
            m.to(self.device)
