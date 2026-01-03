import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

n = 10000
a = torch.ones(n)
b = torch.ones(n)
print(a)

c = torch.zeros(n)
t = time.time()
for i in range(n):
  c[i] = a[i] + b[i]
# print(f'{time.time() - t:5f} sec')

t = time.time()
d = a + b
# print(f"{time.time() - t:5f} sec")

def normal(x, mu, sigma):
  p= 1/math.sqrt(2*math.pi * sigma**2)
  return p * np.exp(-0.5 * (x-mu) ** 2 / sigma**2 )

# Use NumPy again for visualization
x = np.arange(-7, 7, 0.01)
# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
for mu, sigma in params:
  plt.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.gcf().set_size_inches(4.5, 2.5)
plt.show()