import torch
from torchvision import transforms

from lenet import LeNet5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_TRANSFORM = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor()])

BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATA_DIR = './data'

def get_criterion():
    return torch.nn.CrossEntropyLoss()

def print_model_info(model: LeNet5):
    print(f'Epoch: {model.epoch}')
    print(f'Loss history: {model.loss_history}')

def format_loss(value: float):
    return f'{value:.4f}'

def format_accuracy(value: float):
    return f'{value:.2f}'
