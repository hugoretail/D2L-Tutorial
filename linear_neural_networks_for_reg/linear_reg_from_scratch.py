import torch
try:
  from .object_oriented_design_for_impl import Module, HyperParameters, Trainer, add_to_class
  from .synthetic_regression_data import SyntheticRegressionData
except ImportError:  # pragma: no cover
  from object_oriented_design_for_impl import Module, HyperParameters, Trainer, add_to_class
  from synthetic_regression_data import SyntheticRegressionData

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
  l = (y_hat - y) ** 2 / 2
  return l.mean()

class SGD(HyperParameters):
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr
    self.save_hyperparameters()
  
  def step(self):
    with torch.no_grad():
      for param in self.params:
        param -= self.lr * param.grad
  
  def zero_grad(self):
    for param in self.params:
      if param.grad is not None:
        param.grad.zero_()

@add_to_class(LinearRegressionScratch) #@save
def configure_optimizers(self):
  return SGD([self.w, self.b], self.lr)

@add_to_class(Trainer) #@save
def prepare_batch(self, batch):
  return batch

@add_to_class(Trainer) #@save
def fit_epoch(self):
  self.model.train()
  for batch in self.train_dataloader:
    loss = self.model.training_step(self.prepare_batch(batch))
    self.optim.zero_grad()
    loss.backward()
    if self.gradient_clip_val > 0:
      self.clip_gradients(self.gradient_clip_val, self.model)
    self.optim.step()
    self.train_batch_idx += 1
  if self.val_dataloader is None:
    return
  self.model.eval()
  for batch in self.val_dataloader:
    with torch.no_grad():
      self.model.validation_step(self.prepare_batch(batch)) 
    self.val_batch_idx += 1
  
if __name__ == '__main__':
  model = LinearRegressionScratch(2, lr=0.03)
  data = SyntheticRegressionData(w=torch.tensor([2,-3.4]), b=4.2)
  trainer = Trainer(max_epochs=3)
  trainer.fit(model, data)
  
  with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')