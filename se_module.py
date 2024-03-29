from torch import nn
# taken from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

class SELayer(nn.Module):
    def __init__(self, channel, connections):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, connections, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(connections, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
