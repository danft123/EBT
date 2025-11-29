from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError

    def collate_fn(self, batch):
        raise NotImplementedError