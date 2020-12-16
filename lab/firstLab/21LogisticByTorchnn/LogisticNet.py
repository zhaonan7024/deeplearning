from torch import nn

class LogisticNet(nn.Module):
    def __init__(self,n_feature):
        super(LogisticNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
