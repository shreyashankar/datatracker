import torch

class PyTorchTrackedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset
    
    def __getitem__(self, index):
        ret = self.dataset[index]

        if isinstance(ret, list):
            return ret + [index]
        
        if isinstance(ret, tuple):
            return ret + (index, )
        
        return ret, index
    
    def __len__(self):
        return len(self.dataset)