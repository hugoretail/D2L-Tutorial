import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import d2l
import numpy as np
## import gluon # Removed: not needed for PyTorch

# Add the linear_neural_networks_for_reg directory to sys.path for import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../linear_neural_networks_for_reg')))

# Dynamically import object_oriented_design_for_impl from the correct path
import importlib.util

module_name = "object_oriented_design_for_impl"
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../linear_neural_networks_for_reg/object_oriented_design_for_impl.py'))
if not os.path.exists(module_path):
    raise ImportError(f"Module file not found: {module_path}")
spec = importlib.util.spec_from_file_location(module_name, module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load spec for module: {module_name} at {module_path}")
object_oriented_design_for_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(object_oriented_design_for_impl)
DataModule = object_oriented_design_for_impl.DataModule
add_to_class = object_oriented_design_for_impl.add_to_class

class Classifier(DataModule): #@save
  """The base class of Classification models"""
  def validation_step(self,batch):
    Y_hat = self(*batch[:-1])
    self.plot('loss',self.loss(Y_hat,batch[-1]),train=False)
    self.plot('acc',self.accuracy(Y_hat,batch[-1]),train=False)
  
  @add_to_class(DataModule) #@save
  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(),lr=self.lr)

@add_to_class(Classifier) #@save
def accuracy(self,Y_hat,Y,averaged=True):
  Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
  preds = Y_hat.argmax(axis=1).astype(Y.dtype)
  compare = (preds == Y.reshape(-1).astype(np.float32))
  return compare.mean() if averaged else compare

@add_to_class(DataModule) #@save
def get_scratch_params(self):
  params = []
  for attr in dir(self):
    a= getattr(self,attr)
    if isinstance(a, np.ndarray):
      params.append(a)
    if isinstance(a,DataModule):
      params.extend(a.get_scratch_params()) # type: ignore
  return params

@add_to_class(DataModule) #@save
def parameters(self):
  params = self.collect_params()
  return params if isinstance(params,gluon.parameter.ParameterDict) and len(
    params.keys()) else self.get_scratch_params()

  