import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from the_base_classification_model import Classifier
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

from the_image_classification_dataset import FashionMNIST

class SoftmaxRegression(Classifier): #@save
  def __init__(self, num_outputs, lr):
    super().__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))
    
  def forward(self,X):
    return self.net(X)

@add_to_class(Classifier) #@save
def loss(self, Y_hat, Y, averaged=True):
  Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
  Y = Y.reshape((-1,))
  return F.cross_entropy(
    Y_hat, Y, reduction='mean' if averaged else 'none'
  )

def main():
    data = FashionMNIST(batch_size=256)
    model = SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = object_oriented_design_for_impl.Trainer(max_epochs=10)
    trainer.fit(model, data)

if __name__ == "__main__":
    main()