import torch.nn as nn
import torch.nn.functional as F


class GenNet(nn.Module):
    def __init__(self, input_channels=1, input_dim=28):
        super(GenNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        dim = (input_dim - (5 - 1)) // 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        dim = (dim - (5 - 1)) // 2
        self.fcdim = dim
        self.fc1 = nn.Linear(self.fcdim*self.fcdim*16, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.fcdim*self.fcdim*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self, input_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class BigSeedCNN(nn.Module):

    def __init__(self):
        super().__init__()
        d = 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * d * d, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        d = 2
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * d * d)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CFNet(nn.Module):
    def __init__(self):
        super(CFNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
