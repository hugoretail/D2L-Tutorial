import torch

x = torch.arange(4.0, requires_grad=True)
# print(x)
# print(x.grad)
y = 2* torch.dot(x,x)
# print(y)
# print(y.backward())
# print(x.grad)
y.backward()
# print(x.grad == 4 * x)
assert x.grad is not None
x.grad.zero_()
y = x.sum()
y.backward()
# print(x.grad)

x.grad.zero_()
y=x*x
y.backward(gradient=torch.ones(len(y)))
x.grad
print(x.grad)
