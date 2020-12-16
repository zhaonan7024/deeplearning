import torch
from matplotlib import pyplot as plt

n_data = torch.ones(50, 2)  # 数据的基本形态
x1 = torch.normal(2 * n_data, 1)  #
y1 = torch.zeros(50)  # 类型0 shape=(50, 1)
x2 = torch.normal(-2 * n_data, 1)  # shape=(50, 2)
y2 = torch.ones(50)  # 类型1 shape=(50, 1)
# 注意 x, y 数据的数据形式一定要像下面一样(torch.cat是合并数据)
x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
y = torch.cat((y1, y2), 0).type(torch.FloatTensor)



plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0,cmap='RdYlGn')
plt.show()
