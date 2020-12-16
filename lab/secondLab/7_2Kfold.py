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
def evaluate_accuracy(X,y,net,loss):
    acc_sum,n = 0.0,0
    test_l_sum = 0.0
    y_hat = net(X)
    p = y_hat.ge(0.5).float()
    acc_sum += (p.view(p.shape[0]) == y).sum().item()
    n+=y.shape[0]
    l = loss(net(X), y).sum()
    test_l_sum += l.item()
    return acc_sum/n,test_l_sum/n


#定义模型训练函数
def train(net,data,loss,num_epochs,params=None,lr = None,optimizer = None):
    train_1_sum,train_acc_sum,n = 0.0,0.0,0
    for epoch in range(num_epochs):
        X_train = data[0]
        y_train = data[1]
        X_valid = data[2]
        y_valid = data[3]
        y_hat = net(X_train)
        p = y_hat.ge(0.5).float()
        l = loss(y_hat,y_train)
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
        train_acc_sum += (p.view(p.shape[0]) == y_train).sum().item()
        n+=y_train.shape[0]
        test_acc,test_1 = evaluate_accuracy(X_valid,y_valid,net,loss)
    return train_1_sum / n, test_1, train_acc_sum / n, test_acc

lr = 0.01
loss = torch.nn.BCELoss()


#获取K折交叉验证某一折的训练集和验证集
def get_kfold_data(k,i,X,y):
    fold_size = X.shape[0] //k
    val_start = i*fold_size
    if i!=k-1:
        val_end = (i+1)*fold_size
        X_valid,y_valid = X[val_start:val_end],y[val_start:val_end]
        X_train = torch.cat((X[0:val_start],X[val_end:]),dim = 0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:
        X_valid,y_valid = X[val_start:],y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]
    return  X_train,y_train,X_valid,y_valid

def k_fold(k,X_train,y_train):
    train_loss_sum,valid_loss_sum = 0,0
    train_acc_sum,valid_acc_sum = 0,0
    for i in range(k):
        print('第',i+1,'折验证结果')
        data = get_kfold_data(k,i,X_train,y_train)
        train_loss, val_loss, train_acc, val_acc= train(net,data,loss,num_epochs,params,lr)
        print('train_loss: %.8f,val_loss:%.8f,train_acc:%.8f,val_acc:%.8f'%(train_loss,val_loss,train_acc,val_acc))
        train_loss_sum += train_loss
        valid_loss_sum +=val_loss
        train_acc_sum +=train_acc
        valid_acc_sum +=val_acc
    print('最终k折交叉验证结果：')
    print('average train loss :{:.4f},average train accuracy:{:.3f}%'.format(train_loss_sum/k,train_acc_sum/k*100))
    print('average valid loss :{:.4f},average valid accuracy:{:.3f}%'.format(valid_loss_sum/k,valid_acc_sum/k*100))
    return
num_epochs = 30
k = 10
k_fold(k,X,Y)


