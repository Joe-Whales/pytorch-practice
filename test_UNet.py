import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import yaml
from custom_dataset import CustomDataset
from torchvision import transforms

class DoubleConv(nn.Module):
    '''
        Two Convolutional layers with BatchNorm2d and ReLU after each.
        
        Attributes:
        ----------
        in_ch: int
            the number of input channels
        out_ch: int
            the number of output channels
    '''
    def __init__(self, in_ch : int, out_ch : int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    '''
        The UNet model.
        
        Attributes:
        ----------
        in_ch: int
            the number of input channels
        out_ch: int
            the number of output channels
    '''
    def __init__(self, in_ch : int = 7, out_ch : int = 7):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))

        return self.out_conv(d1)

def train_model(model : torch.nn.Module, train_loader : DataLoader, num_epochs : int = 10, learning_rate : float = 0.001, weight_decay : float = 0.0001, device : str = 'cpu', model_path : str = "model.pth"):
    '''
    Trains the given model using the dataloader provided and saves its best state.

    Parameters:
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        The DataLoader for the training data.
    num_epochs : int, optional
        The number of epochs to train the model (default is 10).
    learning_rate : float, optional
        The learning rate for the optimizer (default is 0.001).
    weight_decay : float, optional
        The weight decay (L2 penalty) for the optimizer (default is 0.0001).
    device : str, optional
        The device to use for training, either 'cpu' or 'cuda' (default is 'cpu').
    model_path : str, optional
        The path to save the best model state (default is "model.pth").

    Returns:
    -------
    None
    
    '''
    
    model.to(device)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
            val_loss /= len(train_loader.dataset)
        
        print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}' 
               .format(epoch+1, num_epochs, i+1, total_step, train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Model saved at epoch {epoch}, with val loss: {val_loss}')

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Hyper-parameters
    try:
        with open("baseline_config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        num_epochs = cfg["num_epochs"]
        batch_size = cfg["batch_size"]
        learning_rate = cfg["learning_rate"]
        weight_decay = cfg["weight_decay"]
        data_path = cfg["data_path"]
        data_stats_path = cfg["data_stats_path"]
        model_path = cfg["model_path"]
    except Exception as e:
        print("Error reading config file: \n", e)
        exit()
        
    mean = np.load("train_means.npy")
    std = np.load("train_sdv.npy")
    # print(mean, std)
    mean = torch.tensor(mean).to(device)
    std = torch.from_numpy(std).to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Load data
    train_data = CustomDataset(data_path + "train", transform, (256, 256))
    test_data = CustomDataset(data_path + "val", transform, (256, 256))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Initialize model
    model = UNet()

    # Train model
    train_model(model, train_loader, num_epochs, learning_rate, weight_decay, device)
    
    # Load saved model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test model and compare the loss for each of the labels
    normal_loss = 0.0
    case_1_loss = 0.0
    case_2_loss = 0.0
    normal_count = 0
    case_1_count = 0
    case_2_count = 0
    for images, labels in test_loader:
        outputs = model(images)
        for i in range(len(images)):
            loss = torch.nn.functional.mse_loss(outputs[i], images[i])
            if labels[i] == 0:
                normal_loss += loss.item()
                normal_count += 1
            elif labels[i] == 1:
                case_1_loss += loss.item()
                case_1_count += 1
            elif labels[i] == 2:
                case_2_loss += loss.item()
                case_2_count += 1
    print(f'From {len(test_loader.dataset)} images: Normal count: {normal_count}, Case 1 count: {case_1_count}, Case 2 count: {case_2_count}')
    print(f"Normal loss: {normal_loss}, Case 1 loss: {case_1_loss}, Case 2 loss: {case_2_loss}")
    print(f'Average Normal loss: {normal_loss/normal_count}, Average Case 1 loss: {case_1_loss/case_1_count}, Average Case 2 loss: {case_2_loss/case_2_count}')