import random
import torch
import torch.utils.data as torch_data
try:
  from .object_oriented_design_for_impl import DataModule, add_to_class
except ImportError:  # pragma: no cover
  from object_oriented_design_for_impl import DataModule, add_to_class

class SyntheticRegressionData(DataModule):
  def __init__(self,w,b,noise=0.01,num_train=1000,num_val=1000,
                batch_size=32):
    super().__init__()
    self.save_hyperparameters()
    # The line `n = num_train + num_val` is calculating the total number of data points that will be generated for the synthetic regression dataset. It adds the number of training data points (`num_train`) and the number of validation data points (`num_val`) to determine the total size of the dataset (`n`).
    n = num_train + num_val
    self.X = torch.randn(n, len(w))
    noise = torch.randn(n,1) * noise
    self.y = torch.matmul(self.X, w.reshape((-1,1))) + b + noise
    
if __name__ == '__main__':
  data = SyntheticRegressionData(w=torch.tensor([2,-3.4]), b=4.2)
  X, y = next(iter(data.train_dataloader()))
  print('X shape:', X.shape, '\ny shape:', y.shape)
  print(len(data.train_dataloader()))

# @add_to_class(SyntheticRegressionData)
# def get_dataloader(self, train):
#   if train:
#     indices = list(range(0,self.num_train))
#     random.shuffle(indices)
#   else:
#     indices = list(range(self.num_train, self.num_train+self.num_val))
#   for i in range(0, len(indices), self.batch_size):
#     # The line `batch_indices = torch.tensor(indices[i: i+self.batch_size])` is creating a PyTorch tensor containing a batch of indices for selecting a subset of data points from the dataset.
#     batch_indices = torch.tensor(indices[i: i+self.batch_size])
#     # `yield self.X[batch_indices], self.y[batch_indices]` is a generator expression that yields a batch of input features (`self.X`) and corresponding target labels (`self.y`) based on the batch indices provided.
#     yield self.X[batch_indices], self.y[batch_indices]

# X, y = next(iter(data.train_dataloader()))
# print('X shape:', X.shape, '\ny shape:', y.shape)

@add_to_class(DataModule)
def get_tensorloader(self,tensors,train,indices=slice(0,None)):
  tensors = tuple(a[indices] for a in tensors)
  dataset = torch_data.TensorDataset(*tensors)
  return torch_data.DataLoader(dataset, self.batch_size,
                                      shuffle=train)

@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
  # The line `i = slice(0,self.num_train) if train else slice(self.num_train, None)` is a conditional expression in Python.
  i = slice(0,self.num_train) if train else slice(self.num_train, None)
  return self.get_tensorloader((self.X,self.y), train, i)
