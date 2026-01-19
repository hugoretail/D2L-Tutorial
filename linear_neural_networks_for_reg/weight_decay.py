import torch
from torch import nn
import torch.utils.data as torch_data

try:
  from .object_oriented_design_for_impl import DataModule, add_to_class
except ImportError:  # pragma: no cover
  from object_oriented_design_for_impl import DataModule, add_to_class

class Data(DataModule):
  num_train: int
  num_val: int
  num_inputs: int
  batch_size: int

  def __init__(self, num_train, num_val, num_inputs, batch_size):
    self.save_hyperparameters()
    self.num_train = int(num_train)
    self.num_val = int(num_val)
    self.num_inputs = int(num_inputs)
    self.batch_size = int(batch_size)
    n = num_train + num_val
    self.X = torch.randn(n, num_inputs)
    noise = torch.randn(n,1) * 0.01
    w, b = torch.ones((num_inputs,1)) * 0.01, 0.05
    self.y = torch.matmul(self.X, w) + b + noise

  def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch_data.TensorDataset(*tensors)
    return torch_data.DataLoader(dataset, self.batch_size, shuffle=train)
  
  def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.X, self.y], train, i)