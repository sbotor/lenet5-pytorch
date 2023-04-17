import torch.nn as nn
import torch

_NUM_CLASSES = 10


class LeNet5(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=_NUM_CLASSES)

        self.pool = nn.AvgPool2d(kernel_size=2)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=_NUM_CLASSES)

        self._epochs_trained = 0

    @property
    def epochs_trained(self):
        return self._epochs_trained

    def mark_epoch(self):
        self._epochs_trained += 1

    def forward(self, x: torch.Tensor):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.tanh(self.conv3(x))

        x = x.reshape(x.shape[0], -1)
        x = self.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))  # Not sure

        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str):
        model = LeNet5()
        model.load_state_dict(torch.load(path))

        return model
