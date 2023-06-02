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
        self.activ = nn.Tanh()

        self.epoch = 0

    def forward(self, x: torch.Tensor):
        x = self.activ(self.conv1(x))
        x = self.pool(x)
        x = self.activ(self.conv2(x))
        x = self.pool(x)
        x = self.activ(self.conv3(x))

        x = x.reshape(x.shape[0], -1)
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))

        return x

    @staticmethod
    def save(path: str, model: 'LeNet5', optimizer, loss_history):
        state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_history': loss_history,
            'epoch': model.epoch
        }

        torch.save(state, path)

    @staticmethod
    def load(path: str):
        state = torch.load(path)
        model = LeNet5()

        model.load_state_dict(state['model_state'])
        model.epoch = state['epoch']
        
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(state['optimizer_state'])

        loss_history = state['loss_history']

        return model, optimizer, loss_history
