from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
# from utils.agent_utils import get_dataset
import importlib
from utils.dataset_utils import coll_fn

class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()

        self.config = dataset_param
        mod = importlib.import_module(f"datasets.{self.config.dataset_name}")
        self.dataset = getattr(mod, self.config.dataset_name)(self.config)
        # self.dataset = get_dataset(self.config.dataset_name, self.config)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):
            # Load dataset
            val_length = int(len(self.dataset)*self.config.split_val)
            lengths = [len(self.dataset)-val_length, val_length]
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, lengths)

        if stage == "predict":
            mod = importlib.import_module(f"datasets.{self.config.dataset_name}")
            self.dataset = getattr(mod, self.config.dataset_name)(self.config, train=False)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn
        )
        return val_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            collate_fn=coll_fn,
            pin_memory=True,
        )
        return predict_loader
