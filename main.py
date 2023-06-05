from checkpoints import CheckpointTrainer
from lenet import LeNet5
from utils import get_optimizer

LEARNING_RATE = 0.001
EPOCHS = 5
CP_INTERVAL = 1

def init():
    model = LeNet5()
    optimizer = get_optimizer(model, LEARNING_RATE)
    run_info = {
        'learningRate': LEARNING_RATE,
        'epochs': EPOCHS,
        'chekpointInterval': CP_INTERVAL
    }

    return model, optimizer, run_info


def main():
    model, optimizer, run_info = init()
    runner = CheckpointTrainer(model, optimizer)

    runner.start(EPOCHS, CP_INTERVAL, run_info=run_info)


if __name__ == '__main__':
    main()
