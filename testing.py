import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from lenet import LeNet5

from torcheval.metrics.functional import multiclass_accuracy, multiclass_precision, multiclass_f1_score, multiclass_recall

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

        data_len = len(loader.dataset)
        all_preds = torch.zeros((data_len, LeNet5.NUM_CLASSES))
        all_labels = torch.zeros(data_len, dtype=torch.int64)
        i = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(images)

                proc_count = labels.size(0)
                end_i = i + proc_count

                all_preds[i:end_i, :] = outputs
                all_labels[i:end_i] = labels

                i = end_i

        acc = 100 * multiclass_accuracy(all_preds, all_labels).item()
        print(f'Accuracy: {format_accuracy(acc)}%')

        return all_preds, all_labels

    def _get_loader(self):
        if self._loader:
            return self._loader

        self._dataset = datasets.MNIST(
            './data', train=False, transform=DATASET_TRANSFORM, download=True)

        self._loader = DataLoader(
            self._dataset, batch_size=BATCH_SIZE, shuffle=True)

        return self._loader
    
    @staticmethod
    def getAccuracy(preds: torch.Tensor, labels: torch.Tensor):
        acc = 100 * multiclass_accuracy(preds, labels, num_classes=10, average='macro').item()
        return acc
    
    @staticmethod
    def getPrecision(preds: torch.Tensor, labels: torch.Tensor):
        prec = 100*multiclass_precision(preds, labels, num_classes=10, average='macro').item()
        return prec
    
    @staticmethod
    def getRecall(preds: torch.Tensor, labels: torch.Tensor):
        rec = 100 * multiclass_recall(preds, labels, num_classes=10, average='macro').item()
        return rec

    @staticmethod
    def getF1Score(preds: torch.Tensor, labels: torch.Tensor):
        f1 = 100 * multiclass_f1_score(preds, labels, num_classes=10, average='macro').item()
        return f1

    @staticmethod
    def load(path: str):
        model, _ = LeNet5.load(path)

        return Tester(model)
