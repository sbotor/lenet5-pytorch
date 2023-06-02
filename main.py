from lenet import LeNet5
from utils import Trainer, load_trainer
import torch

LEARNING_RATE = 0.0001

def create_and_train():
    model = LeNet5()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, optim)

    trainer.train(1)

    LeNet5.save('model.pt', model, optim, trainer.loss_history)

def load_and_train():
    trainer = load_trainer('model.pt')

    trainer.train(10)

    LeNet5.save('model.pt', trainer.model, trainer.optimizer, trainer.loss_history)

def print_data():
    model, _, loss_history = LeNet5.load('model.pt')

    print(f'Epoch: {model.epoch}')
    print(f'Loss history: {loss_history}')

def main():
    load_and_train()
    print_data()


if __name__ == '__main__':
    main()
