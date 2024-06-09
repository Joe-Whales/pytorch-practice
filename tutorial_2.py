import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # Data loading
        self.xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        xy = self.xy
        # x is the input data, y is the output data or labels
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers is the number of subprocesses to use for data loading

# Define a simple neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = dataset.x.shape[1]
hidden_size = 5
output_size = 1
model = NeuralNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
            
#torchvision.datasets.MNIST(root='data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
#torchvision.datasets.FashionMNIST(root='data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)