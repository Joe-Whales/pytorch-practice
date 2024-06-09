import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Data preparation
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32, device=device)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32, device=device)
X_test = torch.tensor([5], dtype=torch.float32, device=device)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

# Model definition
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

input_size = n_features
output_size = n_features
model = LinearRegression(input_size, output_size).to(device)

# Print initial prediction
print(f'Prediction before training: f(5) = {model(X_test).item():.5f}')

# Hyperparameters
learning_rate = 0.1
n_iters = 100

# Loss and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# TensorBoard setup
writer = SummaryWriter('runs/linear_regression_experiment_1')

# Training loop
for epoch in range(n_iters):
    model.train()
    
    # Forward pass
    y_pred = model(X)
    l = loss(Y, y_pred)

    # Backward pass and optimization
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log the loss
    writer.add_scalar('Loss/train', l.item(), epoch)
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.3f}')
        # Log weight and bias values
        writer.add_scalar('Weight/w', w[0][0].item(), epoch)
        writer.add_scalar('Bias/b', b.item(), epoch)

# Print final prediction
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

# Close the TensorBoard writer
writer.close()
