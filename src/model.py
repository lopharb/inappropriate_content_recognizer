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


class Model(nn.Module):
    def __init__(self, params, freeze: bool, vgg):
        """
        Initializes the Model class.

        Parameters:
            params (list): A list of parameters for the classifier.
            freeze (bool): A flag indicating whether to freeze the parameters of the extractor.
            vgg (nn.Module): An instance of the VGG model.

        Returns:
            None
        """
        super().__init__()
        self.extractor = vgg.features
        for param in self.extractor.parameters():
            param.requires_grad = not freeze
        self.flatten = nn.Flatten(1, 3)
        self.BN = nn.BatchNorm1d(25088)
        self.classifier = nn.Sequential(*params)

    def forward(self, x):
        features = self.extractor(x)
        flattened = self.flatten(features)
        classes = self.classifier(self.BN(flattened))
        return torch.nn.functional.softmax(classes)


def get_model(pretrained=True, path=None):
    """
    Returns a pretrained model if `pretrained` is `True`, otherwise returns a model with a custom classifier.

    Args:
        pretrained (bool, optional): Whether to load a pretrained model. Defaults to `True`.

    Returns:
        torch.nn.Module: The loaded model or a model with a custom classifier.

    Raises:
        FileNotFoundError: If the pretrained model file is not found.

    Note:
        - If `pretrained` is `True`, the model is loaded from the file 'frozen_MSE_lr0.1_best.pt' and set to evaluation mode.
        - If `pretrained` is `False`, a VGG19 model is instantiated with the weights from ImageNet1K_V1.
        - The custom classifier is defined with a list of linear layers, batch normalization layers, ReLU activation functions, and dropout layers.
        - The model is initialized with the custom classifier, the `freeze` parameter set to `False`, and the VGG19 model as the extractor.
    """
    vgg19 = torchvision.models.vgg19(
        torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    params_list = [nn.Linear(25088, 4096, bias=True),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm1d(4096),
                   nn.Dropout(0.5),
                   nn.Linear(4096, 4096, bias=True),
                   nn.ReLU(inplace=True),
                   nn.BatchNorm1d(4096),
                   nn.Dropout(0.5),
                   nn.Linear(4096, 1000, bias=True),
                   nn.BatchNorm1d(1000),
                   nn.ReLU(inplace=True),
                   nn.Dropout(0.5),
                   nn.Linear(1000, 2, bias=True)]
    if pretrained:
        model = None
        if torch.cuda.is_available():
            model = Model(params=params_list, freeze=False, vgg=vgg19)
            model.load_state_dict(torch.load(path))
            model.eval()
        else:
            model = Model(params=params_list, freeze=False, vgg=vgg19)
            model.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))
            model.eval()
        return model

    model = Model(params_list, freeze=False, vgg=vgg19)
    return model
