import torch
p = torch.normal(0, 0.01, size=(3, 2))
q = torch.normal(0, 0.01, size=(4, 2))
#q的转置
qt = q.t()
#矩阵p与q求内积
print(p)
print(q)
print(qt)

#两个张量的矩阵乘积
t = torch.matmul(p, q.t())
print(t)