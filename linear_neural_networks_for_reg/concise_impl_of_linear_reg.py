import numpy as np
import torch
from torch import nn
from synthetic_regression_data import SyntheticRegressionData
from object_oriented_design_for_impl import Module, add_to_class, Trainer

class LinearRegression(Module):
  def __init__(self,lr):
    super().__init__()
    self.save_hyperparameters()
    self.net = nn.LazyLinear(1)
    self.net.weight.data.normal_(0,0.01)
    self.net.bias.data.fill_(0)
    
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

model = LinearRegression(lr=0.03)
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = Trainer(max_epochs=3)
trainer.fit(model, data)

@add_to_class(LinearRegression) #@save
def get_w_b(self):
  return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

# Helper to access true w and b for error calculation
true_w = torch.tensor([2, -3.4])
true_b = 4.2

print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')

import numpy as np
import matplotlib.pyplot as plt

# Exemple de données (à remplacer par tes résultats)
N_values = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
errors = np.array([0.8, 0.6, 0.45, 0.35, 0.3, 0.25, 0.22, 0.2, 0.19, 0.185, 0.18])

plt.figure()
plt.loglog(N_values, errors, 'o-')  # Échelle log-log
plt.xlabel("Nombre d'échantillons (N)")
plt.ylabel("Erreur d'estimation (MSE)")
plt.title("Erreur vs. Quantité de données (échelle logarithmique)")
plt.grid(True, which="both", ls="--")
plt.show()
