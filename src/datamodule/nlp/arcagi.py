from src.dataset.arcagi import ArcAGIDataset
from src.datamodule.base import BaseDataModule

class ArcAGIDataModule(BaseDataModule):
    def preprocess(self):
        # Custom preprocessing logic for ArcAGI dataset, at the end must define self.train_dataset, self.val_dataset, self.test_dataset lists
        print("Preprocessing ArcAGI dataset...")
        # Get paths for each split from dataset config