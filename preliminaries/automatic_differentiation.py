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
# print(x.grad)

x.grad.zero_()
y = x*x
u = y.detach()
z = u*x
z.sum().backward()
# print(x.grad == u)

x.grad.zero_()
y.sum().backward()
# print(x.grad == 2 * x)

def f(a):
  b = a * 2
  while b.norm() < 1000:
    b*=2
  if b.sum() > 0:
    c=b
  else:
    c = 100*b
  return c

a = torch.randn(size=(),requires_grad=True)
d=f(a)
# print(d.backward())
# print(a.grad==d/a)

import matplotlib.pyplot as plt

x = torch.linspace(-2*torch.pi, 2*torch.pi, 1000, requires_grad=True)
y = torch.sin(x)
y.sum().backward()  # Dérivée de la somme (équivalent à dériver chaque élément)
dy_dx = x.grad      # Récupère f'(x) = cos(x)

plt.plot(x.detach(), y.detach(), label='sin(x)')
plt.plot(x.detach(), dy_dx, label='cos(x) (dérivée)')
plt.legend()
plt.show()
