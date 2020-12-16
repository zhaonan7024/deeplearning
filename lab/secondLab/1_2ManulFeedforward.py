import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

#功能：手动实现前馈神经网络解决二分类任务

#数据集
n_train = 7000*2
n_data = torch.ones(10000, 200)  # 数据的基本形态
x1 = torch.normal(200 * n_data, 1)  # shape=(10000, 200)
y1 = torch.zeros(10000)  # 类型0 shape=(10000, 1)
x2 = torch.normal(-200 * n_data, 1)  # shape=(10000, 200)
y2 = torch.ones(10000)  # 类型1 shape=(10000, 1)
X = torch.cat((x1, x2), 0).type(torch.FloatTensor)
Y = torch.cat((y1, y2), 0).type(torch.FloatTensor)
index = [i for i in range(len(X))]
random.shuffle(index)
X = X[index]
Y = Y[index]
X_train, X_test = X[:n_train, :], X[n_train:, :]
y_train, y_test = Y[:n_train], Y[n_train:]

#批量读取数据
batch_size = 8
dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


#模型参数定义和初始化
num_inputs,num_outputs,num_hiddens = 200,1,256
W1 = torch.tensor(np.random.normal(0,0.01,(num_outputs,num_inputs)),dtype=torch.float)
b1 = torch.zeros(num_outputs,dtype=torch.float)
#W2 = torch.tensor(np.random.normal(0,0.01,(num_outputs,num_hiddens)),dtype=torch.float)
#b2 = torch.zeros(num_outputs,dtype=torch.float)
params = [W1,b1]
for param in params:
    param.requires_grad_(requires_grad=True)

def sigmoid(X):
    return 1.0/(1.0+np.exp(-X))

#定义交叉损失函数
###loss = torch.nn.CrossEntropyLoss()
#loss = torch.nn.MSELoss()
loss = torch.nn.BCELoss()
#定义模型
def net(X):
    X = X.view((-1,num_inputs))
    H = torch.sigmoid(torch.matmul(X,W1.t())+b1)
    return H
#    return torch.matmul(H,W2.t())+b2

#定义随机梯度下降函数
def SGD(params,lr):
    for param in params:
        param.data -=lr*param.grad

#计算模型在某个数据集上的准确率
def evaluate_accuracy(data_iter,net,loss):
    acc_sum,n = 0.0,0
    test_l_sum = 0.0
    for X,y in data_iter:
        y_hat = net(X)
        p = y_hat.ge(0.5).float()
        acc_sum += (p.view(p.shape[0]) == y).sum().item()
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
            p = y_hat.ge(0.5).float()
            l = loss(y_hat,y)
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
            train_acc_sum += (p.view(p.shape[0]) == y).sum().item()
            n+=y.shape[0]
        test_acc,test_1 = evaluate_accuracy(test_iter,net,loss)
        train_loss.append(train_1_sum / n)
        test_loss.append(test_1)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f' % (epoch + 1, train_1_sum / n, train_acc_sum / n, test_acc))
    return train_loss, test_loss

num_epochs = 5
lr = 0.01
batch_size = 5
loss = torch.nn.BCELoss()
train_loss,test_loss=train(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)



#绘制loss曲线
x = np.linspace(0,len(train_loss),len(train_loss))
plt.plot(x,train_loss,label="train_loss",linewidth = 1.5)
plt.plot(x,test_loss,label = "test_loss",linewidth = 1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()


