import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import init
import LinearNet,FlattenLayer
from collections import OrderedDict


# 获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False, download=True, transform=transforms.ToTensor())

# 使用Fashion-MNIST数据集，并设置批量大小为256
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

print(train_iter)
# 定义和初始化模型
num_inputs = 784
num_outputs = 10

#net = LinearNet.LinearNet(num_inputs, num_outputs)

net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer.FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

#softmax和交叉熵损失函数
loss = nn.CrossEntropyLoss()

#定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

#训练模型
num_epochs = 5

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def evaluate_accuracy(data_iter, net):
     acc_sum, n = 0.0, 0
     for X, y in data_iter:
         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
         n += y.shape[0]
     return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)



#PyTorch提供的函数往往具有更好的数值稳定性。可以使用PyTorch更简洁地实现softmax回归。

