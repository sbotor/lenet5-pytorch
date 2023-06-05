from torch import Tensor
import torch.nn as nn
import torch

class ScaledTanh(nn.Tanh):
    def __init__(self, scale: float = 1):
        super().__init__()

        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        input = super().forward(input)
        return input.mul(self.scale)

class LeNet5(nn.Module):
    NUM_CLASSES = 10

    _TANH_SCALE = 1.7159

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=LeNet5.NUM_CLASSES)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ = ScaledTanh(LeNet5._TANH_SCALE)

        self.epoch = 0
        self.loss_history = []

    def forward(self, x: torch.Tensor):
        x = self.activ(self.conv1(x))
        x = self.pool(x)
        x = self.activ(self.conv2(x))
        x = self.pool(x)
        x = self.activ(self.conv3(x))

        x = x.flatten(1)
        x = self.activ(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def create_state(self):
        return {
            'model_state': self.state_dict(),
            'loss_history': self.loss_history,
            'epoch': self.epoch,
            'model_type': self.get_model_type()
        }

    def setup(self, state: dict[str, any]):
        self.epoch = state['epoch']
        self.loss_history = state['loss_history']

    def get_model_type(self):
        return type(self).__name__

class LeNet5ReLU(LeNet5):
    def __init__(self):
        super().__init__()

        self.activ = nn.ReLU()

class LeNet5AvgPool(LeNet5):
    def __init__(self):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class LeNet5ReLUAvgPool(LeNet5):
    def __init__(self):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activ = nn.ReLU()