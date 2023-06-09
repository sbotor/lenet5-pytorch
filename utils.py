import torch
from torchvision import transforms

from lenet import LeNet5, LeNet5ReLU, LeNet5AvgPool, LeNet5ReLUAvgPool

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_TRANSFORM = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor()])

BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATA_DIR = './data'


def get_criterion():
    return torch.nn.CrossEntropyLoss()


def get_optimizer(model: LeNet5, learning_rate: float = None):
    return torch.optim.Adam(model.parameters(), learning_rate or LEARNING_RATE)


def print_model_info(model: LeNet5):
    print(f'Epoch: {model.epoch}')
    print(f'Loss history: {model.loss_history}')


def save_model(path: str, model: LeNet5, optimizer):
    state = model.create_state()
    state['optimizer_state'] = optimizer.state_dict()

    torch.save(state, path)


def load_model(path: str):
    def create_model(type_name: str):
        match type_name:
            case LeNet5.__name__:
                return LeNet5()
            case LeNet5ReLU.__name__:
                return LeNet5ReLU()
            case LeNet5AvgPool.__name__:
                return LeNet5AvgPool()
            case LeNet5ReLUAvgPool.__name__:
                return LeNet5ReLUAvgPool()
            case _:
                raise ValueError(f'Invalid LeNet5 type {type_name}.')

    state = torch.load(path)
    model = create_model(state['model_type'])

    model.load_state_dict(state['model_state'])
    model.setup(state)

    optimizer = get_optimizer(model)
    optimizer.load_state_dict(state['optimizer_state'])

    return model, optimizer


def format_loss(value: float):
    return f'{value:.4f}'


def format_accuracy_percent(value: float):
    return f'{value:.2f}'
