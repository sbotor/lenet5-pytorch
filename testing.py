import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from lenet import LeNet5

from utils import BATCH_SIZE, DATASET_TRANSFORM, DEVICE, format_accuracy, get_criterion


class Tester:
    def __init__(self, model: LeNet5):
        self.model = model

        self._dataset = None
        self._loader = None

        self._criterion = get_criterion()

    def test(self):
        loader = self._get_loader()

        self.model.to(DEVICE)
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(images)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = 100 * correct / total
        print(f'Accuracy: {format_accuracy(acc)}% ({correct}/{total})')

    def _get_loader(self):
        if self._loader:
            return self._loader

        self._dataset = datasets.MNIST(
            './data', train=False, transform=DATASET_TRANSFORM, download=True)

        self._loader = DataLoader(
            self._dataset, batch_size=BATCH_SIZE, shuffle=True)

        return self._loader

    @staticmethod
    def load(path: str):
        model, _ = LeNet5.load(path)

        return Tester(model)
