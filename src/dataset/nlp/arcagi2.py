from src.dataset.base import BaseDataset
import json

class ArcAGI2Dataset(BaseDataset):
    def __init__(self, path_list: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = path_list # list of file paths
        self.data = self.load_data()
    
    def load_data(self):
        # files are json files with keys train and test. Each key maps to a list of dicts with 'input' and 'output' keys each with values as lists of lists of integers
        data = []
        for file_path in self.data_path:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                data.extend(file_data['data'])  # assuming 'data' key contains the list of samples



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Implement item retrieval logic here
        item = self.data[idx]
        return item

    def collate_fn(self, batch):
        # Implement custom collation logic here
        return batch