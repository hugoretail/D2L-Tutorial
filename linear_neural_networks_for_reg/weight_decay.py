import torch
from torch import nn
import torch.utils.data as torch_data
from linear_reg_from_scratch import LinearRegressionScratch
from object_oriented_design_for_impl import DataModule, Trainer, Module
import matplotlib.pyplot as plt

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
  
def l2_penalty(w):
  return (w**2).sum() / 2

class WeightDecayScratch(LinearRegressionScratch):
  def __init__(self, num_inputs, lambd, lr, sigma=0.01):
    super().__init__(num_inputs, lr, sigma)
    self.save_hyperparameters()
  
  def loss(self, y_hat, y):
    return (super().loss(y_hat, y)) + self.lambd * l2_penalty(self.w)

def train_scratch(lambd):
  model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
  model.board.yscale='log'
  trainer.fit(model, data)

  # When running as a plain script (terminal), IPython's display hooks won't
  # open a GUI window; save the figure so you can see it.
  fig = getattr(model.board, 'fig', None)
  if fig is not None:
    out_path = f'weight_decay_lambd_{lambd}.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved plot to {out_path}')

  return ('L2 norm of w:', float(l2_penalty(model.w)))


class WeightDecay(Module):
  def __init__(self, num_inputs, wd, lr):
    super().__init__()
    self.save_hyperparameters()
    self.wd: float = float(wd)
    self.lr: float = float(lr)
    self.net = nn.Linear(int(num_inputs), 1)
    self.net.weight.data.normal_(0, 0.01)
    self.net.bias.data.fill_(0)

  def forward(self, X):
    return self.net(X)

  def loss(self, y_hat, y):
    return nn.MSELoss()(y_hat, y)

  def configure_optimizers(self):
    return torch.optim.SGD(
      [
        {'params': self.net.weight, 'weight_decay': float(self.wd)},
        {'params': self.net.bias},
      ],
      lr=float(self.lr),
    )

if __name__ == '__main__':
  data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
  trainer = Trainer(max_epochs=10)
  # print(train_scratch(0))
  # print(train_scratch(5))
  
  model = WeightDecay(num_inputs=200, wd=3, lr=0.01)
  model.board.yscale='log'
  trainer.fit(model,data)
  print('L2 norm of w:', float(l2_penalty(model.net.weight)))

  fig = getattr(model.board, 'fig', None)
  if fig is not None:
    out_path = 'weight_decay.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved plot to {out_path}')

  # If you have a GUI backend installed (TkAgg/Qt), this will pop a window.
  # Otherwise, rely on the saved PNG.
  try:
    plt.show()
  except Exception:
    pass

  