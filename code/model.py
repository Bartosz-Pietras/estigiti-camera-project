import torch
import torch.nn as nn
import torch.nn.functional as functional


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(60208, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(functional.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 60208)  # -> n, 400
        x = functional.relu(self.fc1(x))  # -> n, 120
        x = functional.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)
        return x
