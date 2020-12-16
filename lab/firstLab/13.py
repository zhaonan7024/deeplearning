import torch
x = torch.ones(1,requires_grad=True)
y1=x*x
with torch.no_grad():
    y2=x*x*x
y3 = y1+y2
y3.backward()
print(y3)
print(x.grad)
#习题解答：在计算x^3的过程中中断了梯度的追踪，导致计算不出梯度，最后结果只计算出了x^2的梯度

