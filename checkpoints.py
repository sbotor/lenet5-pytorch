from math import ceil
import time, torch, os
from lenet import LeNet5
from training import Trainer
from testing import Tester
from csv import writer

class CheckpointTrainer:

    def __init__(self, model: torch.Tensor, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.trainer = Trainer(model, optimizer, True)
        self.tester = Tester(model)

    def start(self, epochs: int, interval: int, silent = False):
        self._epochs = epochs
        self._interval = interval

        self.pathBase = f'checkpoints/{time.strftime("%Y%m%d_%H%M")}'
        os.makedirs(self.pathBase, exist_ok=True)

        iterations = ceil(epochs/interval)

        for i in range(iterations):
            self._loop(i+1, silent)
        

    def _loop(self, it: int, silent = False) -> str:

        if(it*self._interval > self._epochs): #last iteration
            self.trainer.train(it*self._interval - self._epochs)
        else:
            self.trainer.train(self._interval)

        trainedUntil = min(self._epochs, it*self._interval)

        epochCompletness = f"{trainedUntil}_{self._epochs}"
        modelFileName = f"{self.pathBase}/model_{epochCompletness}.pt"
        LeNet5.save(modelFileName, self.trainer.model, self.trainer.optimizer)

        self.tester.model = self.trainer.model
        preds, labels = self.tester.test()

        predPath = f"{self.pathBase}/Predictions_{epochCompletness}.pt"
        torch.save(preds, predPath)
        labelPath = f"{self.pathBase}/Labels_{epochCompletness}.pt"
        torch.save(labels, labelPath)

        if(not silent):
            print(f'Checkpoint {it} | {trainedUntil}/{self._epochs} epochs')

        self._appendMetricsToCSV(preds, labels, trainedUntil)
    
    def _appendMetricsToCSV(self, preds:torch.Tensor, labels:torch.Tensor, epoch:int):
        metricPath = f"{self.pathBase}/Metrics.csv"

        if(not os.path.exists(metricPath)):
            headers = ['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1']
            with open(metricPath, 'x') as file:
                wr = writer(file)
                wr.writerow(headers)

        acc = Tester.getAccuracy(preds, labels)
        rec = Tester.getRecall(preds, labels)
        prec = Tester.getPrecision(preds, labels)
        f1 = Tester.getF1Score(preds, labels)

        fields=[epoch, acc, prec, rec, f1]
        with open(metricPath, 'a') as file:
            wr = writer(file)
            wr.writerow(fields)