import torch
import sys
import os
sys.path.insert(0, '../..')
from d2l.torch import Classifier
from d2l import torch as d2l

# Import add_to_class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../linear_neural_networks_for_reg')))
import importlib.util
module_name = "object_oriented_design_for_impl"
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../linear_neural_networks_for_reg/object_oriented_design_for_impl.py'))
spec = importlib.util.spec_from_file_location(module_name, module_path)
if spec is None or spec.loader is None:
  raise ImportError(f"Could not load spec for module: {module_name} at {module_path}")
object_oriented_design_for_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(object_oriented_design_for_impl)
add_to_class = object_oriented_design_for_impl.add_to_class

X = torch.tensor([[1.0,2.0,3.0], [4.0,5.0,6.0]])
# print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

def softmax(X):
  X_exp = torch.exp(X)
  partition = X_exp.sum(1, keepdim=True)
  return X_exp / partition

X = torch.rand((2,5))
X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))

class SoftmaxRegressionScratch(d2l.Classifier): #type: ignore
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
      super().__init__()
      self.save_hyperparameters()
      self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                      requires_grad=True)
      self.b = torch.zeros(num_outputs, requires_grad=True)
      
    def parameters(self):
      return [self.W, self.b]

@add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
  X = X.reshape((-1, self.W.shape[0]))
  return softmax(torch.matmul(X, self.W) + self.b)

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6], [0.3,0.2,0.5]])
# print(y_hat[[0.,1],y])

def cross_entropy(y_hat,y):
  return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

# print(cross_entropy(y_hat,y))

@add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
  return cross_entropy(y_hat, y)

data = d2l.FashionMNIST(batch_size=256) #type: ignore
data.num_workers = 0
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10, num_gpus=(1 if torch.cuda.is_available() else 0)) #type: ignore
trainer.fit(model, data)

X,y = next(iter(data.val_dataloader()))
preds = model(X).argmax(axis=1)
# print(preds.shape)

wrong = preds.type(y.dtype) != y
X,y,preds = X[wrong], y[wrong], preds[wrong]
labels= [a+'\n'+b for a, b in zip(
  data.text_labels(y), data.text_labels(preds))]
data.visualize([X,y], labels=labels)

