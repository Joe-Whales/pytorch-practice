import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # Force CPU due to small batch size
print('Using device:', device)

# Dataset class
class WineDataset(Dataset):
    def __init__(self, xy):
        # x is the input data, y is the output data or labels
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]).long().squeeze().to(device) - 1  # Convert labels to long and squeeze to 1D
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index].to(device), self.y[index].to(device)
    
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

# Define a neural network model with two hidden layers for classification
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        # no softmax at the end because we are using CrossEntropyLoss
        # for binary classification, you can use sigmoid and BCELoss
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # bin_out = torch.sigmoid(out)  # For binary classification
        return out

input_size = train_dataset.x.shape[1]
hidden_size1 = 10  # Size of the first hidden layer
hidden_size2 = 5   # Size of the second hidden layer
num_classes = 3    # Number of classes
model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Lower learning rate

# Training loop
num_epochs = 100  # Increase number of epochs
total_samples = len(train_dataset)
n_iterations = math.ceil(total_samples / 4)

def evaluate(model):
    # Evaluating the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_true = []
        y_pred = []
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        #print(f'Accuracy on test data: {accuracy:.4f}')
    model.train()  # Set the model back to training mode
    return accuracy

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 5 == 0:
        accuracy = evaluate(model)
        print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, loss = {loss.item():.4f}, accuracy = {accuracy:.4f}')
        
        if accuracy == 1:
            print('Perfect accuracy reached!')
            break
