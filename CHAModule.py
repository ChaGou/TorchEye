import torch.nn as nn
import torch.nn.functional as F


class MyNet1(nn.Module):
    def __init__(self, n_in, n_out):
        super(MyNet1, self).__init__() #调用父类的初始化函数
        self.f1 = nn.Linear(n_in, 2000)
        self.f2 = nn.Linear(2000, 1000)
        self.f3 = nn.Linear(1000, 500)
        self.f4 = nn.Linear(500, 200)
        self.f5 = nn.Linear(200, n_out)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x = self.f1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f4(x)
        x = F.relu(x)
        x = self.f5(x)

        return x


class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__() #调用父类的初始化函数
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.fc1 = nn.Linear(6 * 42 * 42, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 42 * 42)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class MyNet3(nn.Module):
    def __init__(self):
        super(MyNet3, self).__init__() #调用父类的初始化函数
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.fc1 = nn.Linear(6 * 2 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x=x.view(-1,2,6,10)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 6 * 2 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class MyNet4(nn.Module):
    def __init__(self, n_in, n_out):
        super(MyNet4, self).__init__() #调用父类的初始化函数
        self.f1 = nn.Linear(n_in, 960)
        self.f2 = nn.Linear(960, 480)
        self.f3 = nn.Linear(480, n_out)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = self.f1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f3(x)
        return x
