import numpy as np
import torch
from torch import nn
from synthetic_regression_data import SyntheticRegressionData
from object_oriented_design_for_impl import Module, add_to_class, Trainer

class LinearRegression(Module):
  def __init__(self, lr, num_inputs=None):
    super().__init__()
    self.save_hyperparameters()
    # Prefer a non-lazy layer when input dimensionality is known.
    if num_inputs is None:
      self.net = nn.LazyLinear(1)
    else:
      self.net = nn.Linear(int(num_inputs), 1)
    self.net.weight.data.normal_(0,0.01)
    self.net.bias.data.fill_(0)

  def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
    
@add_to_class(LinearRegression)
def forward(self,X):
  return self.net(X)

@add_to_class(LinearRegression)
def loss(self, y_hat, y):
  fn = nn.MSELoss()
  return fn(y_hat, y)

@add_to_class(LinearRegression)
def configure_optimizers(self):
  return torch.optim.SGD(self.parameters(), self.lr)


if __name__ == '__main__':
  model = LinearRegression(lr=0.03, num_inputs=2)
  data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
  trainer = Trainer(max_epochs=3)
  trainer.fit(model, data)

  w, b = model.get_w_b()
  true_w = torch.tensor([2, -3.4])
  true_b = 4.2
  print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
  print(f'error in estimating b: {true_b - b}')
