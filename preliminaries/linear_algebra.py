import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x**y)

x = torch.arange(3)
# print(x)

# print(x[2])
# print(len(x))
# print(x.shape)

A = torch.arange(6).reshape(3,2)
# print(A)
# print(A.T)

A = torch.tensor([[1,2,3], [2, 0,4], [3,4,5]]) #symétrique
# print(A)
# print(A==A.T)

# print(torch.arange(24).reshape(2,3,4))
A = torch.arange(6, dtype=torch.float32).reshape(2,3)
B = A.clone()
# print(A, B, A+B, A*B)

a=2
X=torch.arange(24).reshape(2,3,4)
# print(f"{a + X} \n {(a*X).shape}")

