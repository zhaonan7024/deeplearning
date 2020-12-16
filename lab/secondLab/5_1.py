import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#在多分类任务实验中分别手动实现dropout
#多分类数据集问题
train_dataset = torchvision.datasets.MNIST(root='~/Datasets/MNIST',train=True,download=True,transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='~/Datasets/MNIST',train=False,transform=transforms.ToTensor())

#批量读取数据
batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
test_iter = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)

#手动实现dropout,当使用dropout时，前馈神经网络隐藏层中的隐藏单元ℎ_𝑖有一定概率被丢弃掉
def dropout(X,drop_prob):
    X = X.float()
    #检查丢弃概率是否在0到1之间
    assert  0<=drop_prob <=1
    keep_prob = 1-drop_prob
    if keep_prob == 0:
        return  torch.zeros_like(X)
    #生成mask矩阵向量
    mask = (torch.rand(X.shape)< keep_prob).float()
    return mask *X/keep_prob

#模型参数定义和初始化
num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 784,10,256,256
W1 = torch.tensor(np.random.normal(0,0.01,(num_hiddens1,num_inputs)),dtype=torch.float)
b1 = torch.zeros(num_hiddens1,dtype=torch.float)
W2 = torch.tensor(np.random.normal(0,0.01,(num_hiddens2,num_hiddens1)),dtype=torch.float)
b2 = torch.zeros(num_hiddens2,dtype=torch.float)
W3 = torch.tensor(np.random.normal(0,0.01,(num_outputs,num_hiddens1)),dtype=torch.float)
b3 = torch.zeros(num_outputs,dtype=torch.float)
params = [W1,b1,W2,b2,W3,b3]
for param in params:
    param.requires_grad_(requires_grad=True)

#定义激活函数 ReLU
def relu(X):
    return torch.max(input = X,other=torch.tensor(0.0))

#定义交叉损失函数
loss = torch.nn.CrossEntropyLoss()
#定义模型
#使用dropout网络模型，两个隐藏层的丢弃率分别是0.2,0.5
drop_probe1,drop_probe2 = 0.2,0.5
def net(X,is_Trainning = True):
    X = X.view((-1,num_inputs))
    H1 = (torch.matmul(X,W1.t())+b1).relu()
    if is_Trainning:
        H1 = dropout(H1,drop_probe1)
    H2 = (torch.matmul(H1,W2.t())+b2).relu()
    if is_Trainning:
        H2 = dropout(H2,drop_probe2)
    return torch.matmul(H2,W3.t())+b3

#定义随机梯度下降函数
def SGD(params,lr):
    for param in params:
        param.data -=lr*param.grad


#计算模型在某个数据集上的准确率
def evaluate_accuracy(data_iter,net,loss):
    acc_sum,n = 0.0,0
    test_l_sum = 0.0
    for X,y in data_iter:
        acc_sum+=(net(X,is_Trainning=False).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
        l = loss(net(X), y).sum()
        test_l_sum += l.item()
    return acc_sum/n,test_l_sum/n

#定义模型训练函数
def train(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr = None,optimizer = None):
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        train_1_sum,train_acc_sum,n = 0.0,0.0,0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                SGD(params,lr)
            else:
                optimizer.step()
            train_1_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
        test_acc,test_1 = evaluate_accuracy(test_iter,net,loss)
        train_loss.append(train_1_sum / n)
        test_loss.append(test_1)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f' %(epoch+1,train_1_sum/n,train_acc_sum/n,test_acc))
    return train_loss, test_loss

num_epochs = 5
lr = 0.1
train_loss,test_loss=train(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)


#绘制loss曲线
x = np.linspace(0,len(train_loss),len(train_loss))
plt.plot(x,train_loss,label="train_loss",linewidth = 1.5)
plt.plot(x,test_loss,label = "test_loss",linewidth = 1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
