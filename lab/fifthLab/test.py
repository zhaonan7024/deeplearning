import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=2):
        super(LSTM, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


num_time_steps = 1920
num_time_steps_test = 960
input_size = 1
hidden_size = 8
output_size = 1
lr = 0.01
is_gpu = torch.cuda.is_available()
if is_gpu:
    device = torch.device("cuda")
    print("Use GPU.")
else:
    device = torch.device("cpu")
    print("Use CPU.")
model = LSTM(2, 4, 1, 2).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros(1, 1, hidden_size).to(device)
volume_train = np.load(open("./出租车流量预测数据集/NYC-stdn/volume_test.npz", "rb"))["volume"]
volume_test = np.load(open("./出租车流量预测数据集/NYC-stdn/volume_test.npz", "rb"))["volume"]
train_data = volume_train[:, 0][:, 0][:, 0]
test_data = volume_test[:, 0][:, 0][:, 0]
max_value = np.max(train_data)  # 获得最大值
min_value = np.min(train_data)  # 获得最小值
scalar = max_value - min_value  # 获得间隔数量
train_data = list(map(lambda x: x / scalar, train_data))  # 归一化
max_value = np.max(test_data)  # 获得最大值
min_value = np.min(test_data)  # 获得最小值
scalar = max_value - min_value  # 获得间隔数量
test_data = list(map(lambda x: x / scalar, test_data))  # 归一化


def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# 创建好输入输出
data_X, data_Y = create_dataset(train_data)
data_test_X, data_test_Y = create_dataset(test_data)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X))
test_size = int(len(test_data))
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_test_X
test_Y = data_test_Y


train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X).float().to(device)
train_y = torch.from_numpy(train_Y).float().to(device)
test_x = torch.from_numpy(test_X).float().to(device)

# 开始训练
for e in range(3000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {}'.format(e + 1, loss.item()))

model = model.eval() # 转换成测试模式

test_X = test_X.reshape(-1, 1, 2)
test_X = torch.from_numpy(test_X).float().to(device)
var_data = Variable(test_X)
pred_test = model(var_data) # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.cpu().numpy()

# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(test_data, 'b', label='real')
plt.legend(loc='best')
plt.show()
