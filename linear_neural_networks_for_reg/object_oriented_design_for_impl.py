import time
import numpy as np
import torch
from torch import nn
import collections
import inspect
from IPython import display
import matplotlib.pyplot as plt

def add_to_class(Class): #@save
  """Register functions as methods in created class."""
  def wrapper(obj):
    setattr(Class, obj.__name__, obj)
  return wrapper

class A:
  def __init__(self):
    self.b = 1

@add_to_class(A)
def do(self):
  print('Class attribute "b" is ', self.b)
  
 # a.do()

# ---

class HyperParameters: #@save
    """The base class of hyperparameters"""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplementedError

@add_to_class(HyperParameters)  #@save
def save_hyperparameters(self, ignore=[]):
    """Save function arguments into class attributes."""
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("Unable to inspect the caller frame to save hyperparameters.")
    frame = frame.f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k:v for k, v in local_vars.items()
                    if k not in set(ignore+['self']) and not k.startswith('_')}
    for k, v in self.hparams.items():
        setattr(self, k, v)
        

class B(HyperParameters):
    def __init__(self, a, b, c):
        # Set attributes first, then call save_hyperparameters
        self.a = a
        self.b = b
        # self.c is intentionally not set
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

if __name__ == '__main__':
  # Demo calls
  a = A()
  a.do()
  b = B(a=1,b=2,c=3)

class ProgressBoard(HyperParameters):
  """The Board that plots data points in animation."""
  def __init__(self, xlabel=None, ylabel=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         ls=['-','--','-.',':'], colors=['C0','C1','C2','C3'],
         fig=None, axes=None, figsize=(3.5,2.5), display=True):
    # Explicitly set all attributes to avoid missing attribute errors
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.xlim = xlim
    self.ylim = ylim
    self.xscale = xscale
    self.yscale = yscale
    self.ls = ls
    self.colors = colors
    self.fig = fig
    self.axes = axes
    self.figsize = figsize
    self.display = display
    self.save_hyperparameters()

  def draw(self, x, y, label, every_n=1):
    Point = collections.namedtuple('Point', ['x', 'y'])
    if not hasattr(self, 'raw_points'):
      self.raw_points = collections.OrderedDict()
      self.data = collections.OrderedDict()
    if label not in self.raw_points:
      self.raw_points[label] = []
      self.data[label] = []
    points = self.raw_points[label]
    line = self.data[label]
    points.append(Point(x, y))
    if len(points) != every_n:
      return
    mean = lambda x: sum(x) / len(x)
    line.append(Point(mean([p.x for p in points]),
              mean([p.y for p in points])))
    points.clear()
    if not self.display:
      return
    # use_svg_display()  # Commented out, as not defined in this context
    if self.fig is None:
      self.fig = plt.figure(figsize=self.figsize)
    plt_lines, labels = [], []
    for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
      plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                    linestyle=ls, color=color)[0])
      labels.append(k)
    axes = self.axes if self.axes else plt.gca()
    if self.xlim: axes.set_xlim(self.xlim)
    if self.ylim: axes.set_ylim(self.ylim)
    if self.xlabel: axes.set_xlabel(self.xlabel)
    if self.ylabel: axes.set_ylabel(self.ylabel)
    axes.set_xscale(self.xscale)
    axes.set_yscale(self.yscale)
    axes.legend(plt_lines, labels)
    display.display(self.fig)
    display.clear_output(wait=True)
  
if __name__ == '__main__':
  board = ProgressBoard('x')
  for x in np.arange(0,10,0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)

class Module(nn.Module, HyperParameters):
  def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
    super().__init__()
    self.save_hyperparameters()
    self.board = ProgressBoard()
    
  def loss(self, y_hat, y):
    raise NotImplementedError
  
  def forward(self, X):
    assert hasattr(self, 'net'), 'Neural network is defined'
    return self.net(X)

  def plot(self, key, value, train):
    """Plot a point in animation."""
    assert hasattr(self, 'trainer'), 'Trainer is not inited'
    self.board.xlabel = 'epoch'
    if train:
      x = float(self.trainer.train_batch_idx) / float(self.trainer.num_train_batches)
      n = float(self.trainer.num_train_batches) / float(self.plot_train_per_epoch)
    else:
      x = float(self.trainer.epoch) + 1.0
      n = float(self.trainer.num_val_batches) / float(self.plot_valid_per_epoch)
    self.board.draw(x, value.detach().cpu().numpy(),
            ('train_' if train else 'val_') + key,
            every_n=int(n))

  def training_step(self, batch):
    l = self.loss(self(*batch[:-1]), batch[-1])
    self.plot('loss', l, train=True)
    return l

  def validation_step(self, batch):
    l = self.loss(self(*batch[:-1]), batch[-1])
    self.plot('loss', l, train=False)
    return l

  def configure_optimizers(self):
    raise NotImplementedError
    
class DataModule(HyperParameters):
  def __init__(self, root='../data', num_workers=4):
    self.save_hyperparameters()
    
  def get_dataloader(self,train):
    raise NotImplementedError
  
  def train_dataloader(self):
    return self.get_dataloader(train=True)
  
  def val_dataloader(self):
    return self.get_dataloader(train=False)

class Trainer(HyperParameters):
  """The base class for training models with data."""
  def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    assert num_gpus == 0, 'No GPU support yet'

  def prepare_data(self, data):
    self.train_dataloader = data.train_dataloader()
    self.val_dataloader = data.val_dataloader()
    self.num_train_batches = len(self.train_dataloader)
    self.num_val_batches = len(self.val_dataloader) if self.val_dataloader is not None else 0
    
  def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    self.model = model

  def fit(self, model, data):
    self.prepare_data(data)
    self.prepare_model(model)
    self.optim = model.configure_optimizers()
    for epoch in range(self.max_epochs):
      self.epoch = epoch
      self.train_batch_idx = 0
      self.val_batch_idx = 0
      self.fit_epoch()

  def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
      return
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > grad_clip_val:
      for param in params:
        param.grad[:] *= grad_clip_val / norm

  def fit_epoch(self):
    raise NotImplementedError