from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import yaml

# Define Custom Dataset
class CustomDataset(Dataset):
    '''
    A custom dataset class for loading and preprocessing data.

    Attributes:
    ----------
    root_dir : str
        The root directory where the dataset is stored.
    transform : Callable, optional
        A function/transform to apply to the samples.
    resize_dim : Tuple[int, int], optional
        The dimensions to resize the samples to.
    data : List[np.ndarray]
        A list to store the loaded data samples.
    labels : List[int]
        A list to store the labels for the data samples.
    class_names : List[str]
        A list of class names derived from the root directory.
        
    '''
    def __init__(self, root_dir, transform=None, resize_dim=None):
        self.root_dir = root_dir
        self.transform = transform
        self.resize_dim = resize_dim
        self.data = []
        self.labels = []
        self.class_names = os.listdir(root_dir)
        self.load_data()

    def load_data(self):
        try:
            for i, class_name in enumerate(self.class_names):
                class_dir = os.path.join(self.root_dir, class_name)
                for file in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file)
                    self.data.append(np.load(file_path))
                    self.labels.append(i)
        except Exception as e:
            print(f'Error loading data at {self.root_dir}: \n', e)
            exit()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx : int):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.resize_dim:
            sample = cv2.resize(sample, self.resize_dim)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
    
    def get_class_name(self, idx : int):
        return self.class_names[idx]



def test_loaders(train_data_loader, test_data_loader, num_images=4):
    '''
    Tests the data loaders by visualizing some images.

    Parameters:
    ----------
    train_data_loader : DataLoader, optional
        The DataLoader for the training data.
    test_data_loader : DataLoader, optional
        The DataLoader for the test data.
    num_images : int, optional
        The number of images to visualize (default is 4).
        
    '''
    if train_data_loader:
        # Loop through the 7 layers and print them on a grid in plt
        count = 0
        for image, label in train_data_loader:
            image, label = image[0], label[0]
            print(train_data_loader.dataset.get_class_name(label.numpy()))
            plt.figure(figsize=(15, 4))
            for i in range(7):
                plt.subplot(1, 7, i+1)
                plt.imshow(image[:, :, i], cmap='gray')
                plt.axis('off')
            plt.show()
            if count > num_images:
                break
            count+=1

    if test_data_loader:
        # Print an image from the test set with label 0 or 1
        count = 0
        cases = []
        for image, label in test_data_loader:
            image, label = image[0], label[0]
            if label.numpy() in cases:
                continue
            cases.append(label.item())
            print(test_data_loader.dataset.get_class_name(label.numpy()))
            plt.figure(figsize=(15, 4))
            for i in range(7):
                plt.subplot(1, 7, i+1)
                plt.imshow(image[:, :, i], cmap='gray')
                plt.axis('off')
            plt.show()
            if count > num_images:
                break
            count+=1

if __name__ == "__main__":
    try:
        with open("baseline_config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        batch_size = cfg["batch_size"]
        data_path = cfg["data_path"]
        data_stats_path = cfg["data_stats_path"]
    except Exception as e:
        print("Error reading config file: \n", e)
        exit()

    # Load data
    train_dataset = CustomDataset(root_dir=data_path + "train", resize_dim=(30, 30))
    test_dataset = CustomDataset(root_dir=data_path + "val")

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    test_loaders(train_data_loader, None, 10)