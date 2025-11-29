import hydra
import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class BaseDataModule(L.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def preprocess(self):
        pass

    def setup(self, stage: str | None = None):
        self.preprocess()

        if stage == "fit" or stage is None:
            ...

        if stage == "test" or stage is None:
            ...
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, 'collate_fn') else None
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.test_dataset.collate_fn if hasattr(self.test_dataset, 'collate_fn') else None
        )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_data_module(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup()
    print(f"Train dataset: {type(dm.train_dataset)}")
    print(f"Val dataset: {type(dm.val_dataset)}")
    print(f"Test dataset: {type(dm.test_dataset)}")

if __name__ == "__main__":
    run_data_module()