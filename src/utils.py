import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image
import json
from tqdm.notebook import tqdm
from itertools import product

IMAGE_SIZE = 224
BATCH_SIZE = 8
CLASS_NAMES = {
    'safe': [0, 1],
    'inappropriate': [1, 0]
}

preprocessors = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, root, transform, mode) -> None:
        """
        Initializes the ImageDataset class.

        Parameters:
            root (str): The root directory of the dataset.
            transform (callable): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
            mode (str): The mode of the dataset. Can be 'train', 'test', or 'valid'.

        Returns:
            None

        Raises:
            AssertionError: If the number of labels is not equal to the number of files.
        """
        self.mode = mode
        self.transforms = transform
        self.files = list()
        self.labels = list()
        image_set_name = 'train'
        if self.mode == 'test':
            image_set_name = 'test'
        if self.mode == 'valid':
            image_set_name = 'valid'

        dir = os.path.join(root, image_set_name)
        for class_dir in os.listdir(dir):
            path_to_all_calss_files = os.path.join(dir, class_dir)
            for filename in os.listdir(path_to_all_calss_files):
                full_path = os.path.join(path_to_all_calss_files, filename)
                if os.path.isfile(full_path):
                    self.files.append(full_path)
                    self.labels.append(CLASS_NAMES[class_dir.lower()])

        assert len(self.labels) == len(
            self.files), "each file should have a label"

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and labels of the item.
                - img (torch.Tensor): The image tensor.
                - labels (torch.Tensor): The labels tensor.
        """
        img = Image.open(self.files[index % len(self.files)])
        img = self.transforms(img)
        labels = self.labels[index % len(self.labels)]

        return (img, torch.tensor(labels, dtype=torch.float32))

    def __len__(self):
        return len(self.files)


def plot_history(history, title: str):
    plt.figure(figsize=(12, 8))
    epochs = len(history['train'])
    plt.plot(range(epochs), history['train'], label='train')
    plt.plot(range(epochs), history['val'], label='val')
    plt.title(title)
    plt.legend()
    plt.show()


def show_image(img):
    if isinstance(img, torch.Tensor):
        plt.imshow(img.view(-1, IMAGE_SIZE, IMAGE_SIZE).permute(1,
                                                                2, 0).detach().cpu().numpy())
    elif isinstance(img, str):
        img = Image.open(img)
        plt.imshow(img)
    else:
        plt.imshow(img)
    plt.show()
