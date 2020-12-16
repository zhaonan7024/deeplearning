import torch
import numpy as np
from itertools import zip_longest
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import pandas as pd
import math
from torch.utils.data import Dataset as dataset
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
#GRU

#获取数据集
data1 = np.load('data/实验4-数据/高速公路传感器数据/PEMS04/PEMS04.npz',allow_pickle=True)
data1 = data1['data']
data1 = data1[:, :, 0]
data2 = np.load('data/实验4-数据/高速公路传感器数据/PEMS07/PEMS07.npz',allow_pickle=True)
data2 = data2['data']
data3 = np.load('data/实验4-数据/高速公路传感器数据/PEMS08/PEMS08.npz',allow_pickle=True)
data3 = data3['data']
print(data1.shape)
print(data2.shape)
print(data3.shape)

#模型实现
#torch实现
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        out, self.state = self.rnn(inputs, state)
        outs = []
        for time_step in range(out.size(1)):  # calculate output for each time step
            outs.append(self.dense(out[:, time_step, :]))
        return state, torch.stack(outs, dim=1)


#固定滑动窗口
def sliding_window(seq,window_size):
    result = []
    for i in range(len(seq)-window_size):
        result.append(seq[i:i+window_size])
    return  result

#划分长序列、短序列
train_set,test_set = [],[]
full_seq = data1
full_len =full_seq.shape[0]
train_seq,test_seq = full_seq[:int(full_len*0.8)], full_seq[int(full_len*0.8):]
#训练集、测试集
train_set+= sliding_window(train_seq,window_size=13)
train_set = np.concatenate(train_set, axis=1).transpose()
test_set+=sliding_window(test_seq,window_size=13)
test_set = np.concatenate(test_set, axis=1).transpose()
print(train_seq.shape,test_seq.shape)
train_set,test_set = np.array(train_set),np.array(test_set)
#,test_set = (item[~np.isnan(item).any(axis=1)] for item in (train_set,test_set))
print(train_set.shape,test_set.shape)

device = torch.device("cpu")
lr = 1e-2 # 注意调整学习率
gru_layer = nn.GRU(input_size=1, hidden_size=32)
model = RNNModel(gru_layer, 1).to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)

def mape(y_true,y_pred):
    y_true, y_pred = y_true.detach().numpy(), y_pred.detach().numpy()
    non_zero_index = (y_true>0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true-y_pred)/y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape)*100
#读取batch
def next_batch(data,batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length/batch_size)
    for batch_index in range(num_batches):
        start_index = batch_size * batch_size
        end_index = min((batch_size +1) *batch_size,data_length)
        yield  data[start_index:end_index]
loss_log = []
mape__log = []
rmse_log = []
mae_log = []
train_batches = 0
num_epoch = 1
state = None

for epoch in range(num_epoch):
    for batch in next_batch(shuffle(train_set),batch_size=64):
        batch = torch.from_numpy(batch).float()
        x,label = batch[:,:12],batch[:,-1]
        hidden, out = model(batch.unsqueeze(-1), state)
        prediction = out[:,-1,:].squeeze(-1)
        loss = loss_func(prediction,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.detach().cpu().numpy().tolist())
        train_batches += 1
        if train_batches %2000 == 0 :
            # 测试函数
            with torch.no_grad():
                test_num = len(test_set)
                total_loss = 0
                total_mape_score = 0
                total_rmse = 0
                total_mae = 0
                for batch in next_batch(test_set,batch_size=64):
                    batch = torch.from_numpy(batch).float()
                    x,label = batch[:,:12],batch[:,-1]
                    hidden, out = model(x.unsqueeze(-1), state)
                    prediction = out[:,-1,:].squeeze(-1)
                    # 指标评价参考：https://blog.csdn.net/weixin_42497252/article/details/102903466
                    loss = loss_func(prediction, label)  # 前向传播
                    total_loss += loss.item()  # 累计样本的loss
                    mape_score = mape(label, prediction)  #模型评价指标mape
                    total_mape_score += mape_score.item()
                    rmse = np.sqrt(mean_squared_error(label, prediction)) #模型评价指标均方根误差
                    total_rmse+=rmse
                    mae = mean_absolute_error(label, prediction) #模型评价指标平均绝对误差
                    total_mae+=mae
                loss = total_loss / test_num
                mape_score = total_mape_score / test_num
                rmse_score = total_rmse/test_num
                mae_score = total_mae/test_num
                loss_log.append(loss)
                mape__log.append(mape_score)
                rmse_log.append(rmse_score)
                mae_log .append(mae_score)
                print(
                      f"loss:{loss:.4f}   "
                      f"mape_score:{mape_score:.4f}   "
                      f"rmse_score:{rmse_score:.4f}   "
                      f"mae_score:{mae_score:.4f}  "
                )

def plt_result(data,ylable):
    #绘制loss曲线
    x = np.linspace(0,len(data),len(data))
    plt.plot(x,data,label=ylable,linewidth = 1.5)
    plt.xlabel("Num of batches")
    plt.ylabel(ylable)
    plt.legend()
    plt.show()

plt_result(loss_log,'loss')
plt_result(mape__log,'map')
plt_result(rmse_log,'rmse')
plt_result(mae_log,'mae')

