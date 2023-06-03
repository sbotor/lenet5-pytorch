from lenet import LeNet5
from testing import Tester
from training import Trainer
from utils import LEARNING_RATE, print_model_info
import torch

MODEL_PATH = 'model.pt'

def _train_and_save(trainer: Trainer):
    trainer.train(5)

    LeNet5.save(MODEL_PATH, trainer.model, trainer.optimizer)

def create_and_train():
    model = LeNet5()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, optim)
    
    _train_and_save(trainer)

def load_and_train():
    trainer = Trainer.load(MODEL_PATH)

    _train_and_save(trainer)

def test():
    tester = Tester.load(MODEL_PATH)
    tester.test()

def print_data():
    model, _ = LeNet5.load(MODEL_PATH)
    print_model_info(model)

def main():
    load_and_train()
    #create_and_train()
    test()
    print_data()


if __name__ == '__main__':
    main()
