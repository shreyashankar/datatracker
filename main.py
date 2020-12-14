import argparse
import numpy as np
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torchvision

from datatracker import DataTracker, PyTorchTrackedDataset

# Parse arguments
parser = argparse.ArgumentParser(description='Parameters.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--input_dim', type=int, default=784)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--output_dim', type=int, default=10)
args = parser.parse_args()
RANDOM_SEED = args.seed
NUM_EPOCHS = args.num_epochs
INPUT_DIM = args.input_dim
HIDDEN_DIM = args.hidden_dim
OUTPUT_DIM = args.output_dim

# Set random seed
torch.manual_seed(RANDOM_SEED)

# Load dataset and transforms
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.5,), (0.5,)),
                                            ])
train_set = torchvision.datasets.MNIST(
    '.data/train', download=True, train=True, transform=transform)
test_set = torchvision.datasets.MNIST(
    '.data/test', download=True, train=False, transform=transform)

train_set = PyTorchTrackedDataset(train_set)
test_set = PyTorchTrackedDataset(test_set)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=True)

# Create model, loss function, and optimizer
# model = torch.nn.Sequential(torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
#                             torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#                             torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
#                             torch.nn.LogSoftmax(dim=1))
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

dt = DataTracker()
dt.add_tracker('loss', lambda loss: loss)

def train(epoch):
    """Run an iteration of training and return the loss and accuracy."""
    model.train()
    total_loss = 0
    num_correct = 0
    for data, target, id in train_loader:
        # data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        num_correct += pred.eq(target.data.view_as(pred)).long().sum()

        individual_losses = torch.nn.NLLLoss(reduction='none')(output, target)
        dt.track(loss=individual_losses.detach().numpy(), label=target.detach().numpy(), iter=epoch, id=id.detach().numpy())

    print('largest losses')
    print(dt.get_largest('loss', 50))
    # print('largest gains in losses')
    # print(dt.get_largest_gain('loss', 50))
    return total_loss, float(num_correct / len(train_set))


def test():
    """Return the loss and accuracy on the test set."""
    model.eval()
    total_loss = 0
    num_correct = 0
    for data, target, _ in test_loader:
        # data = data.view(data.shape[0], -1)
        output = model(data)
        loss = loss_fn(output, target)
        total_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        num_correct += pred.eq(target.data.view_as(pred)).long().sum()
    return total_loss, float(num_correct / len(test_set))


# Train and test
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch}')
    train_loss, train_acc = train(epoch)
    print(f'\tTrain loss: {train_loss}')
    print(f'\tTrain accuracy: {train_acc}')
    test_loss, test_acc = test()
    print(f'\tTest loss: {test_loss}')
    print(f'\tTest accuracy: {test_acc}')

dt.dump_results()