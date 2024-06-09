import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dataset class
class WineDataset(Dataset):
    def __init__(self, xy):
        # x is the input data, y is the output data or labels
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

# Load data
data = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)

# Normalize the data
scaler = StandardScaler()
data[:, 1:] = scaler.fit_transform(data[:, 1:])

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create datasets
train_dataset = WineDataset(train_data)
test_dataset = WineDataset(test_data)

# Create DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=0)

# Define a neural network model with two hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

input_size = train_dataset.x.shape[1]
hidden_size1 = 10  # Size of the first hidden layer
hidden_size2 = 5   # Size of the second hidden layer
output_size = 1
model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Lower learning rate

# Training loop
num_epochs = 100  # Increase number of epochs
total_samples = len(train_dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 50 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, loss = {loss.item():.4f}')

# Evaluating the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_true = []
    y_pred = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(outputs.numpy().flatten())
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Absolute Error on test data: {mae:.4f}')
    print(f'Mean Squared Error on test data: {mse:.4f}')
