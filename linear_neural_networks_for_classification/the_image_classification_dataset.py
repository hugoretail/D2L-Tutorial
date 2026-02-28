import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import d2l

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

class FashionMNIST(DataModule):
  """The Fashion-MNIST dataset."""
  def __init__(self,batch_size=64,resize=(28,28)):
    super().__init__()
    self.save_hyperparameters()
    self.num_workers = 0  # Fix for Windows multiprocessing issues
    trans = transforms.Compose([transforms.Resize(resize),
                                transforms.ToTensor()])
    self.train = torchvision.datasets.FashionMNIST(
      root=self.root, train=True, transform=trans, download=True)
    self.val = torchvision.datasets.FashionMNIST(
      root=self.root, train=False, transform=trans, download=True)

data = FashionMNIST(resize=(32,32))
# print(len(data.train), len(data.val))

# print(data.train[0][0].shape)

@add_to_class(FashionMNIST) #@save
def text_labels(self, indices):
  """Return text labels."""
  labels = ['t-shirt','trouser','pullover','dress','coat',
            'sandal','shirt','sneaker','bag','ankle boot']
  # The line `return [labels[int(i)] for i in indices]` is creating a list of text labels corresponding to the indices provided.
  return [labels[int(i)] for i in indices]

@add_to_class(FashionMNIST) #@save
def get_dataloader(self, train):
  data = self.train if train else self.val
  return DataLoader(data, self.batch_size, shuffle=train,
                    num_workers=self.num_workers)

X, y = next(iter(data.train_dataloader()))
# print(X.shape, X.dtype, y.shape, y.dtype)

tic = time.time()
for X,y in data.train_dataloader():
  continue
f'{time.time() - tic:.2f} sec'

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
  """Plot a list of images."""
  import matplotlib.pyplot as plt
  from matplotlib import gridspec
  figsize = (num_cols * scale, num_rows * scale)
  fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
  axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
  for i, (ax, img) in enumerate(zip(axes, imgs)):
      img_np = img.detach().cpu().numpy()
      if img_np.ndim == 3 and img_np.shape[0] == 1:
          img_np = img_np.squeeze(0)
      ax.imshow(img_np, cmap='gray')
      ax.axis('off')
      if titles and i < len(titles):
          ax.set_title(titles[i], fontsize=8)
  # Hide unused axes
  for ax in axes[len(imgs):]:
      ax.axis('off')
  plt.tight_layout()
  plt.show()

@add_to_class(FashionMNIST) #@save
def visualize(self, batch, nrows=1,ncols=8,labels=[]):
  # `X , y = batch` is unpacking the `batch` variable into two separate variables `X` and `y`. This is commonly used in Python to assign values from an iterable (in this case, `batch`) to multiple variables at once. In this specific context, `batch` likely contains a pair of data and labels, where `X` will represent the data (images) and `y` will represent the corresponding labels.
  X , y = batch
  if not labels:
    # The line `labels = self.text_labels(y)` is calling the `text_labels` method of the `FashionMNIST` class instance (`self`) and passing the `y` variable as an argument. This method is responsible for converting the numerical labels (`y`) into their corresponding text labels. In this case, the `text_labels` method returns a list of text labels based on the numerical labels provided in the `y` variable.
    labels = self.text_labels(y)
  show_images(X.squeeze(1),nrows,ncols,titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)