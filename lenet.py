from torch import Tensor
import torch.nn as nn
import torch

_TANH_SCALE = 1.7159

class LeNet5(nn.Module):
    NUM_CLASSES = 10

    class _ScaledTanh(nn.Tanh):
        def __init__(self, scale: float = 1):
            super().__init__()

            self.scale = scale

        def forward(self, input: Tensor) -> Tensor:
            input = super().forward(input)
            return input.mul(self.scale)

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=LeNet5.NUM_CLASSES)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ = self._ScaledTanh(_TANH_SCALE)

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

    @staticmethod
    def save(path: str, model: 'LeNet5', optimizer,):
        state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_history': model.loss_history,
            'epoch': model.epoch
        }

        torch.save(state, path)

    @staticmethod
    def load(path: str):
        state = torch.load(path)
        model = LeNet5()

        model.load_state_dict(state['model_state'])
        model.epoch = state['epoch']
        model.loss_history = state['loss_history']

        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(state['optimizer_state'])

        return model, optimizer
