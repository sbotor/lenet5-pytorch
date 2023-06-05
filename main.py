from checkpoints import CheckpointTrainer
from lenet import LeNet5, LeNet5ReLU, LeNet5AvgPool, LeNet5ReLUAvgPool
from utils import get_optimizer, print_model_info

LEARNING_RATES = (0.1, 0.01, 0.001, 0.0001)

EPOCHS = 10
INTERVAL = 1

METRICS_PATH = 'metrics.csv'

def create_info(learning_rate: int, model: LeNet5):
    return {
        'learningRate': learning_rate,
        'epochs': EPOCHS,
        'chekpointInterval': INTERVAL,
        'modelType': type(model).__name__
    }


def create_models():
    yield LeNet5()
    yield LeNet5ReLU()
    yield LeNet5AvgPool()
    yield LeNet5ReLUAvgPool()


def run_tests():
    for lr in LEARNING_RATES:
        for model in create_models():
            optimizer = get_optimizer(model, lr)
            runner = CheckpointTrainer(model, optimizer, METRICS_PATH)
            runner.start(EPOCHS, INTERVAL, run_info=create_info(lr, model))
            print_model_info(model)


def main():
    run_tests()


if __name__ == '__main__':
    main()
