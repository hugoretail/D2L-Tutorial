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

A = X.numpy()
B = torch.from_numpy(A)
# print(type(A), type(B))

a = torch.tensor([3.5])
# print(a, a.item(), float(a), int(a))

# Exercises

X = torch.arange(12,dtype=torch.float32).reshape(3,4)
Y = torch.tensor([[4,5,6],[22,2,1],[4,2,1],[2,2,1]]).reshape(3,4)
# print(X.shape, Y.shape)
# print(X)
# print(Y)
# print(X == Y, X < Y, X > Y)

a = torch.arange(1, 6, dtype =torch.float32).reshape((5, 1))
b = torch.arange(1, 3).reshape((1, 2))
# OK