import torch
from lenet import LeNet5
from utils import DATA_DIR, DEVICE, DATASET_TRANSFORM, BATCH_SIZE, format_loss, get_criterion, load_model
from torchvision import datasets
from torch.utils.data import DataLoader


class Trainer:
    class _EpochInfo:
        def __init__(self):
            self.current = 0
            self.total = 0

            self.step = 0
            self.total_steps = 0

            self.model_epoch = 0

    def __init__(self, model: LeNet5, optimizer: torch.optim.Optimizer, silent=False):
        self.model = model
        self.optimizer = optimizer

        self.epoch = self._EpochInfo()
        self.epoch.model_epoch = model.epoch

        self.silent = silent
        self.print_freq = 300

        self._dataset = None
        self._loader = None

        self._criterion = get_criterion()

    def train(self, epochs: int):
        if (epochs < 0):
            raise ValueError(f'Invalid epoch count ({epochs})')

        loader = self._get_loader()

        self.model.to(DEVICE)
        self.model.train()

        self.epoch.total_steps = len(loader)
        dataset_len = len(loader.dataset)

        self.epoch.total = epochs
        self.epoch.current = 0

        for _ in range(epochs):
            running_loss = 0.0

            self.epoch.current += 1
            self.epoch.model_epoch += 1

            for step, (images, labels) in enumerate(loader):
                self.epoch.step = step + 1

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self._criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                loss.backward()
                self.optimizer.step()

                self._print_step_end(loss)

            self.model.epoch += 1
            epoch_loss = running_loss / dataset_len
            self.model.loss_history.append(epoch_loss)
            self._print_epoch_end(epoch_loss)

    def get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _print_step_end(self, loss):
        if self.silent or self.print_freq < 1:
            return

        if (self.epoch.step) % self.print_freq == 0:
            epoch = self.epoch
            print(
                f'Epoch: {epoch.current}/{epoch.total} ({self.epoch.model_epoch} total). Step: {epoch.step}/{epoch.total_steps}. Loss: {format_loss(loss.item())}')

    def _print_epoch_end(self, epoch_loss):
        if self.silent:
            return

        epoch = self.epoch
        print(
            f'*** Finished epoch: {epoch.current}/{epoch.total} ({self.model.epoch} total). Epoch loss: {format_loss(epoch_loss)}')
        print('---------')

    def _get_loader(self):
        if self._loader:
            return self._loader

        self._dataset = datasets.MNIST(
            DATA_DIR, train=True, transform=DATASET_TRANSFORM, download=True)

        self._loader = DataLoader(
            self._dataset, batch_size=BATCH_SIZE, shuffle=True)

        return self._loader

    @staticmethod
    def load(path: str):
        model, optim = load_model(path)

        return Trainer(model, optim)
