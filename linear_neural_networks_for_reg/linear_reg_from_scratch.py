import torch
import sys
sys.path.append('..')
import d2l
from object_oriented_design_for_impl import *

class LinearRegressionScratch(Module):
  def __init__(self, num_inputs, lr, sigma=0.01):
    super().__init__()
    self.save_hyperparameters()
    self.w = torch.normal(0, sigma, (num_inputs,1), requires_grad=True)
    self.b = torch.zeros(1, requires_grad=True)

@add_to_class(LinearRegressionScratch)
def forward(self, X):
  return torch.matmul(X, self.w) + self.b

@add_to_class(LinearRegressionScratch)
def loss(self, y_hat, y):
  l = (y_hat - y) ** 2 / b
  return l.mean()

