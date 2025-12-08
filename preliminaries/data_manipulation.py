import torch

x = torch.arange(12, dtype=torch.float32)
# print(x.numel())
# print(x.shape)
# print(x.reshape(2,6))
# print(x.reshape(4,-1))
zeros = torch.zeros((2,3,4))
# print(zeros)
ones = torch.ones((3,2,2))
# print(ones)
random = torch.randn(7,8)
# print(random)
lst = torch.tensor([[2,4,2,3],[3,1,1,2],[3,2,1,4]])
# print(lst)
X = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(X[1,2])

# print(torch.exp(X))

x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
# print(x+y, x-y, x*y, x/y, x**y)

X = torch.arange(12,dtype=torch.float32).reshape(3,4)
# ...

tf = torch.tensor([[True, False, True],[False, True, False]])
# print(tf.sum())

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
# print(a + b)

# Z = torch.zeros_like(Y)

