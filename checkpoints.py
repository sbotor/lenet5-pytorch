import time, torch, os
from lenet import LeNet5
from training import Trainer
from testing import Tester
from csv import writer

class CheckpointTrainer:

    def __init__(self, epochs: int, interval: int):
        self.epochs = epochs
        self.interval = interval

    def start(self, modelPath:str, silent = False):
        filePath = "checkpoints/" + time.strftime("%Y%m%d_%H%M")+"/"
        metricPath = filePath + "Metrics.csv"
        headers = ['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1']
        os.makedirs(os.path.dirname(metricPath), exist_ok=True)
        with open(metricPath, 'w') as file:
            wr = writer(file)
            wr.writerow(headers)

        iterations = self.epochs//self.interval
        if(self.epochs%self.interval != 0):
            iterations+=1

        for i in range(iterations):
            modelPath = self.loop(i+1, modelPath, filePath, silent)
        

    def loop(self, it: int, modelPath: str, filePath: str, silent = False) -> str:
        trainer = Trainer.load(modelPath)
        trainer.silent = True

        if(it*self.interval > self.epochs): #last iteration
            trainer.train(it*self.interval - self.epochs)
        else:
            trainer.train(self.interval)

        trainedTo = min(self.epochs, it*self.interval)

        epochCompletness = "_" + str(trainedTo) + "_" + str(self.epochs)
        fileName = filePath + "model" + epochCompletness + ".pt"
        LeNet5.save(fileName, trainer.model, trainer.optimizer)

        tester = Tester(trainer.model)
        preds, labels = tester.test()

        predPath = filePath + "Predictions" + epochCompletness + ".pt"
        torch.save(preds, predPath)

        if(not silent):
            print(f'Checkpoint {it} | {trainedTo}/{self.epochs} epochs')

        acc = Tester.getAccuracy(preds, labels)
        rec = Tester.getRecall(preds, labels)
        prec = Tester.getPrecision(preds, labels)
        f1 = Tester.getF1Score(preds, labels)

        metricPath = filePath + "Metrics.csv"
        fields=[trainedTo, acc, prec, rec, f1]
        with open(metricPath, 'a') as file:
            wr = writer(file)
            wr.writerow(fields)

        return fileName