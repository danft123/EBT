import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.arcagi2 import ArcAGI2Dataset


from src.datamodule.base import BaseDataModule

class ArcAGI2DataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def preprocess(self):
        # Custom preprocessing logic for ArcAGI2 dataset, at the end must define self.train_dataset, self.val_dataset, self.test_dataset lists
        print("Preprocessing ArcAGI2 dataset...")
        # Get paths for each split from dataset config
        # Paths are available in self.hparams.dataset.train and self.hparams.dataset.test
        
    
    def setup(self, stage: str | None = None):
        self.preprocess()

        if stage == "fit" or stage is None:
            

def main():
    dm = ArcAGI2DataModule(batch_size=32, num_workers=4, dataset={
        'train': 'data/arcagi2/training/',
        'test': 'data/arcagi2/evaluation/'
    })
    dm.setup('fit')
    print(f"Train dataset: {type(dm.train_dataset)}")
    print(f"Val dataset: {type(dm.val_dataset)}")
    dm.setup('test')
    print(f"Test dataset: {type(dm.test_dataset)}")
    for batch in dm.train_dataloader():
        print(batch)
        break

if __name__ == "__main__":
    main()