"""
The goal of this file is to download the data locally
Imagenet is already present in supercloud, so it will not be downloaded here
"""

import os
import torch
from pathlib import Path
from torchvision import datasets, transforms, models

from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test"
}

LABEL_FILES = {"places365":"data/categories_places365_clean.txt",
               "imagenet":"data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cub":"data/cub_classes.txt"}

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data



def download_dataset(dataset_name, train=True, root_dir=Path.cwd() / 'data' / 'datasets', preprocess=None):
    """
    Downloads the specified dataset and returns a PyTorch dataset object.

    Args:
        dataset_name (str): Name of the dataset to download. Must be one of 'cifar10', 'cifar100', or 'places365'.
        train (bool): Whether to download the training set (True) or test set (False). Default is True.
        root_dir (str): Root directory to save the downloaded dataset. Default is 'data'.

    Returns:
        PyTorch dataset object.
    """
    if dataset_name == 'cifar10':
        data = datasets.CIFAR10(root=os.path.join(root_dir, 'cifar10'), train=train, download=True, transform=preprocess)
    elif dataset_name == 'cifar100':
        data = datasets.CIFAR100(root=os.path.join(root_dir, 'cifar100'), train=train, download=True, transform=preprocess)
    elif dataset_name == 'places365':
        split = 'train-standard' if train else 'val'
        try:
            data = datasets.Places365(root=os.path.join(root_dir, 'places365'), split=split, small=True, download=True, transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.join(root_dir, 'places365'), split=split, small=True, download=False, transform=preprocess)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Must be one of 'cifar10', 'cifar100', or 'places365'.")

    return data


def download_all_datasets():
    """
    A script downloading the different datasets locally

    Please note that there is no transform operation
    """
    download_dataset("cifar10", train=True)
    download_dataset("cifar10", train=False)
    download_dataset("cifar100", train=True)
    download_dataset("cifar100", train=False)
    download_dataset("places365", train=True)
    download_dataset("places365", train=False)



if __name__ == "__main__":
    download_all_datasets()