import torch
#𝐫初始化一个𝟏 × 𝟑的矩阵 𝑴和
m = torch.rand(1,3)
n = torch.rand(2,1)

 # 减法形式一
print(m - n)
# 减法形式二
print(torch.sub(m, n))
# 减法形式三，inplace（原地操作）
m.sub_(n)
print(m)

#第三种方式发生了报错，当对两个形状不同的矩阵按元素运算时，可能会触发广播（broadcasting）机制：先适当复制元素使这两个NDArray形状相同后再按元素运算。直接原地操作导致形状不同运行报错