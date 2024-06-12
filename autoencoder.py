import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

transforms = transforms.ToTensor()
mnist_data = datasets.MNIST(root='../../data', train=True, transform=transforms, download=True)
dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)

data_iter = iter(dataloader)
images, labels = next(data_iter)
print(torch.min(images), torch.max(images))
print(images.shape)

class Linear_Autoencoder(nn.Module):
    def __init__(self):
        super(Linear_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Conv_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7), # 7x7 -> 1x1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    # note: MaxPool2d and ConvTranspose2d are used to downsample and upsample the image, respectively
    # Can use MaxUnpool2d to reverse MaxPool2d
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Note: [-1, 1] -> tanh, [0, 1] -> sigmoid

model = Conv_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
num_epochs = 9
outputs = []
for epoch in range(num_epochs):
    for (img, _) in dataloader:
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))

for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().to("cpu").numpy()
    recon = outputs[k][2].detach().to("cpu").numpy()
    for i, item in enumerate(imgs):
        if i >= 9:
            break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])
    
    for i, item in enumerate(recon):
        if i >= 9:
            break
        plt.subplot(2, 9, 9+i+1)
        # item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])
    plt.show()
