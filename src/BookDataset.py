import torch
from torch.utils.data import Dataset

class BookDataset(Dataset):
  def __init__(self, data): self.data = data

  def __len__(self): return len(self.data)

  def __getitem__(self, idx):
    x, y = self.data[idx]
    return {
      "input_ids": torch.tensor(x),
      "attention_mask": torch.ones_like(torch.tensor(x)),
      "labels": torch.tensor(y)
    }