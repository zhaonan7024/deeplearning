import torch
import numpy as np
import torch.nn as nn
import math
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

#获取数据集
train_path = r"./出租车流量预测数据集/NYC-stdn/volume_test.npz"
test_path = r"./出租车流量预测数据集/NYC-stdn/volume_test.npz"

volume_train=np.load(open(train_path, "rb"))["volume"]
volume_test=np.load(open(test_path, "rb"))["volume"]
volume_train = volume_train[:, :, :,0]
volume_train.shape =  960,200
volume_test = volume_test[:, :, :,0]
volume_test.shape =  960,200
#shape=(1920*10*20*2) 代表有1920个时间段，10*20个区域，2个特征分别为区域的入流量与出流量

#模型实现
#构造一个含单隐藏层、隐藏单元个数为256的循环神经网络层rnn_layer
input_size = 1
hidden_size = 32
rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size)
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, output_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.output_size = output_size
        self.dense = nn.Linear(self.hidden_size, output_size)
        self.state = None

    def forward(self, inputs,state):
        #r_out, state = self.rnn(inputs, state)
        out, self.state = self.rnn(inputs, state)
        #output = self.dense(out)
        #output = torch.stack(output, dim=1)
        #return output, self.state
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
train_seq = volume_train
test_seq = volume_test
#训练集、测试集
train_set+= sliding_window(train_seq,window_size=6)
train_set = np.concatenate(train_set, axis=1).transpose()
test_set+=sliding_window(test_seq,window_size=6)
test_set = np.concatenate(test_set, axis=1).transpose()
train_set,test_set = np.array(train_set),np.array(test_set)

device = torch.device("cpu")
batch_size = 64
seq_len = 6
output_size = 1
model = RNNModel(rnn_layer=rnn_layer,output_size=output_size)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100
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
num_epoch = 50
state = None
score_log = []
trained_batches = 0

for epoch in range(num_epoch):
    for batch in next_batch(shuffle(train_set),batch_size=64):
        batch = torch.from_numpy(batch).float()
        x,label = batch[:,:6],batch[:,-1]
        hidden,out = model(batch.unsqueeze(-1),state)
        prediction = out[:,-1,:].squeeze(-1)
        loss = loss_func(prediction,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.detach().cpu().numpy().tolist())
        train_batches += 1

def lastTest():
    with torch.no_grad():
        test_num = len(test_set)
        total_loss = 0
        total_mape_score = 0
        total_rmse = 0
        total_mae = 0
        for batch in next_batch(test_set, batch_size=64):
            batch = torch.from_numpy(batch).float()
            x, label = batch[:, :6], batch[:, -1]
            hidden, out = model(x.unsqueeze(-1), state)
            prediction = out[:, -1, :].squeeze(-1)
            # 指标评价参考：https://blog.csdn.net/weixin_42497252/article/details/102903466
            loss = loss_func(prediction, label)  # 前向传播
            total_loss += loss.item()  # 累计样本的loss
            mape_score = mape(label, prediction)  # 模型评价指标mape
            total_mape_score += mape_score.item()
            rmse = np.sqrt(mean_squared_error(label, prediction))  # 模型评价指标均方根误差
            total_rmse += rmse
            mae = mean_absolute_error(label, prediction)  # 模型评价指标平均绝对误差
            total_mae += mae
        rmse_score = total_rmse / test_num
        mae_score = total_mae / test_num
        rmse_log.append(rmse_score)
        mae_log.append(mae_score)
        print(            f"rmse_score:{rmse_score:.4f}   "
            f"mae_score:{mae_score:.4f}  "        )

#最终评估
lastTest()

