import json
from math import ceil
import time
import datetime
import torch
import os
from lenet import LeNet5
from training import Trainer
from testing import Tester
from csv import writer

from utils import save_model


class CheckpointTrainer:

    _METRICS_HEADERS = ('Timestamp', 'LR', 'Type', 'Epoch',
                        'Accuracy', 'Precision', 'Recall', 'F1')

    def __init__(self, model: LeNet5, optimizer: torch.optim.Optimizer, metrics_path: str = None):
        self.model = model
        self.optimizer = optimizer
        self.trainer = Trainer(model, optimizer, True)
        self.tester = Tester(model)

        self.metrics_path = metrics_path
        self.metrics_epoch_offset = model.epoch

        self.store_tensors = True

    def start(self, epochs: int, interval: int, silent=False, run_info: dict[str, any] = None):
        self._epochs = epochs
        self._interval = interval

        self.pathBase = f'checkpoints/{time.strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.pathBase, exist_ok=True)

        iterations = ceil(epochs/interval)

        self._store_run_info(run_info)

        for i in range(iterations):
            self._loop(i+1, silent)

    def _loop(self, it: int, silent=False) -> str:

        if (it*self._interval > self._epochs):  # last iteration
            self.trainer.train(it*self._interval - self._epochs)
        else:
            self.trainer.train(self._interval)

        trainedUntil = min(self._epochs, it * self._interval)

        epochCompletness = f"{trainedUntil}_{self._epochs}"
        modelFileName = f"{self.pathBase}/model_{epochCompletness}.pt"
        save_model(modelFileName, self.trainer.model, self.trainer.optimizer)

        self.tester.model = self.trainer.model
        preds, labels = self.tester.test()

        self._store_tensors(preds, labels, epochCompletness)

        if (not silent):
            print(f'Checkpoint {it} | {trainedUntil}/{self._epochs} epochs')

        self._appendMetricsToCSV(preds, labels, trainedUntil)

    def _appendMetricsToCSV(self, preds: torch.Tensor, labels: torch.Tensor, epoch: int):
        metricPath = self.metrics_path or f"{self.pathBase}/Metrics.csv"

        if (not os.path.exists(metricPath)):
            with open(metricPath, 'x', newline='') as file:
                wr = writer(file)
                wr.writerow(CheckpointTrainer._METRICS_HEADERS)

        acc = Tester.getAccuracy(preds, labels)
        rec = Tester.getRecall(preds, labels)
        prec = Tester.getPrecision(preds, labels)
        f1 = Tester.getF1Score(preds, labels)

        actual_epoch = self.metrics_epoch_offset + epoch
        now = datetime.datetime.now()
        fields = (now.isoformat(), self.trainer.get_learning_rate(),
                  self.model.get_model_type(), actual_epoch, acc, prec, rec, f1)
        with open(metricPath, 'a', newline='') as file:
            wr = writer(file)
            wr.writerow(fields)

    def _store_run_info(self, run_info: dict[str, any]):
        if not run_info:
            return

        file_path = f'{self.pathBase}/info.json'
        with open(file_path, 'x') as f:
            json.dump(run_info, f)

    def _store_tensors(self, preds: torch.Tensor, labels: torch.Tensor, epoch_completeness: str):
        if not self.store_tensors:
            return

        predPath = f"{self.pathBase}/Predictions_{epoch_completeness}.pt"
        torch.save(preds, predPath)

        labelPath = f"{self.pathBase}/Labels_{epoch_completeness}.pt"
        torch.save(labels, labelPath)
