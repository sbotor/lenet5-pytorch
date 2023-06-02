import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from lenet import LeNet5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_DATASET_TRANSFORM = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor()])

_BATCH_SIZE = 64

def load_trainer(path):
    model, optim, history = LeNet5.load(path)

    return Trainer(model, optim, history)


class Trainer:
    def __init__(self, model: LeNet5, optimizer: torch.optim.Optimizer, loss_history: list[float] = []):
        self.model = model
        self.optimizer = optimizer
        self.loss_history = loss_history
        
        self._dataset = None
        self._loader = None

        self._criterion = nn.CrossEntropyLoss()

        self._print_freq = 300

    def train(self, epochs: int):
        loader = self._get_loader()

        self.model.to(DEVICE)
        self.model.train()

        total_steps = len(loader)
        dataset_len = len(loader.dataset)

        for epoch in range(epochs):
            running_loss = 0.0

            for i, (images, labels) in enumerate(loader):
                images = images.to(DEVICE)
                labels.to(DEVICE)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self._criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                loss.backward()
                self.optimizer.step()

                if (i + 1) % self._print_freq == 0:
                    print(f'Epoch: {epoch + 1}/{epochs}. Step: {i + 1}/{total_steps}. Loss: {loss.item()}')

            self.model.epoch += 1
            epoch_loss = running_loss / dataset_len
            self.loss_history.append(epoch_loss)
            
            print(f'*** Finished epoch: {epoch + 1}/{epochs} ({self.model.epoch} total). Epoch loss: {epoch_loss:.4f}')
            print('---------')

    def _get_loader(self):
        if self._loader:
            return self._loader

        self._dataset = datasets.MNIST(
            './data', train=True, transform=_DATASET_TRANSFORM, download=True)

        self._loader = DataLoader(
            self._dataset, batch_size=_BATCH_SIZE, shuffle=True)

        return self._loader
