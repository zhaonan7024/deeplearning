import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

n_data = torch.ones(50,2)
x1 = torch.normal(2 * n_data, 1)  # shape=(10000, 200)
y1 = torch.zeros(50,1)  # 类型0 shape=(10000, 1)
x2 = torch.normal(-2 * n_data, 1)  # shape=(10000, 200)
y2 = torch.ones(50,1)  # 类型1 shape=(10000, 1)
x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
y = torch.cat((y1, y2), 0).type(torch.FloatTensor)

#读取数据
def data_iter(batch_size,features,lables):
    num_examples = len(features)
    indices = list(range(num_examples))
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),lables.index_select(0,j)
data = data_iter(1,x,y)


w = torch.zeros([2,1],requires_grad=True,dtype=torch.float32)
b = torch.zeros([1],requires_grad=True,dtype=torch.float32)

def logreg(xx,w,b):
    mid = torch.mm(xx,w)+b
    r = 1/(1+torch.exp(mid))
    return r

loss = torch.nn.BCELoss()

def SGD(params,lr,batch_size):
    for param in params:
        param.data -=lr*param.grad/batch_size

lr = 0.01
num_epochs = 10
batch_size = 1
net = logreg
epochs = []
train_loss = []
acc_sum =[]
for epoch in range(num_epochs):
    cor = 0
    for xx, yy in data_iter(batch_size,x,y):
        out = logreg(xx,w,b)
        p = out.ge(0.5).float()
        cor = cor+(p==yy)
  #      print('yy',yy)
        l = loss(out,yy)
        l.backward()
        SGD([w,b],lr,batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    print('cor[0]', cor)
    acc = cor[0].item()/batch_size
    print(acc)