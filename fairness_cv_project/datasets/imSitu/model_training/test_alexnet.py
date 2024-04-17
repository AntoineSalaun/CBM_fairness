import argparse
import os
import json
from pathlib import Path
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import sys
sys.path.insert(1, str(Path.cwd()))

from fairness_cv_project.methods.label_free_cbm.src.utils import data_utils
from fairness_cv_project.methods.label_free_cbm.src.models import cbm
from fairness_cv_project.methods.label_free_cbm.src.plots import plots


class SingleFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_class=0):
        self.img_dir = Path(img_dir)
        self.img_paths = list(self.img_dir.glob('*.jpg'))  # adjust as necessary if you have other file types
        self.transform = transform
        self.target_class = target_class

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)  # This returns a PIL Image
        if self.transform:
            img = self.transform(img)
        return img, self.target_class


# Load the model
def load_cbm_model(load_dir, device):
    cbm_model = cbm.load_cbm(load_dir, device)
    cbm_model.to(device)
    cbm_model.eval()
    return cbm_model

def load_alexnet_model(path_alexnet_model, device):
    target_model = models.alexnet()
    target_model.classifier[6] = nn.Linear(4096, 2)
    state_dict = torch.load(path_alexnet_model, map_location='cpu')
    target_model.load_state_dict(state_dict)
    target_model = target_model.to(device)

    target_model.eval() 
    return target_model

def main(path_model, is_cbm, path_test_dataset, device):
    if is_cbm:
        target_model = load_cbm_model(path_model, device)
    else:
        target_model = load_alexnet_model(path_model, device)

    # Define your transform
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    categories = ['eating', 'phoning'] #put by alphabetical order
    genders = ['male', 'female']
    
    accuracies = {}

    for category in categories:
        accuracies[category] = {}
        for gender in genders:
            # Get the data
            folder_name = f"{category}_{gender}"
            dataset = SingleFolderDataset(f'{path_test_dataset}/{folder_name}', transform=data_transforms, target_class=0 if      category==categories[0] else 1)  # or 1 for 'phoning'

            # Create the dataloader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

            correct = 0
            total = 0

            with torch.no_grad():
                for data in dataloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = target_model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on {folder_name} test images: {100  * correct / total }')
            accuracies[category][gender] = 100 * correct / total 

    return accuracies


def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df

if __name__ == '__main__':
    """
    path_cbm = {
        'Dataset Balanced': {
            'GPT Concepts': 'saved_models/imSitu/phoning_eating/phoning_eating_balanced_phoning_eating_filtered_new_cbm_2023_07_06_17_39',
            'GPT Concepts with gender': 'saved_models/imSitu/phoning_eating/phoning_eating_balanced_phoning_eating_filtered_gender_cbm_2023_07_06_17_38',
            'Augmented concepts': 'saved_models/imSitu/phoning_eating/phoning_eating_balanced_phoning_eating_augmented_filtered_cbm_2023_07_06_17_39',
            'Augmented concepts with gender': 'saved_models/imSitu/phoning_eating/phoning_eating_balanced_phoning_eating_augmented_filtered_gender_cbm_2023_07_06_17_38',
        },
        'Dataset Imbalanced 1': {
            'GPT Concepts': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_1_phoning_eating_filtered_new_cbm_2023_07_06_17_39',
            'GPT Concepts with gender': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_1_phoning_eating_filtered_gender_cbm_2023_07_06_17_38',
            'Augmented concepts': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_1_phoning_eating_augmented_filtered_cbm_2023_07_06_17_40',
            'Augmented concepts with gender': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_1_phoning_eating_augmented_filtered_gender_cbm_2023_07_06_17_38',
        },
        'Dataset Imbalanced 2': {
            'GPT Concepts': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_2_phoning_eating_filtered_new_cbm_2023_07_06_17_39',
            'GPT Concepts with gender': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_2_phoning_eating_filtered_gender_cbm_2023_07_06_17_38',
            'Augmented concepts': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_2_phoning_eating_augmented_filtered_cbm_2023_07_06_17_40',
            'Augmented concepts with gender': 'saved_models/imSitu/phoning_eating/phoning_eating_imbalanced_2_phoning_eating_augmented_filtered_gender_cbm_2023_07_06_17_38',
        },
    }
    path_model = Path('saved_models/imSitu/phoning_eating/imbalanced_phoning_female_case_2.pt')
    path_test_dataset = Path('data/datasets/imSitu/data/phoning_eating/human_images/test')
    device = 'cuda'
    is_cbm = True

    results = {}

    for dataset, value in path_cbm.items():
        results[dataset] = {}
        for concept_set, path in value.items():
            results[dataset][concept_set] = {}
            path_model = path
            accuracies = main(path_model, is_cbm, path_test_dataset, device)
            results[dataset][concept_set] = accuracies
    
    print(results)

    df = nested_dict_to_df(results)
    df.to_csv('saved_models/imSitu/phoning_eating/results_cbm.csv')
    """
    path_models = [Path('saved_models/imSitu/phoning_eating/balanced.pt'),
                   Path('saved_models/imSitu/phoning_eating/imbalanced_phoning_male_case_1.pt'),
                   Path('saved_models/imSitu/phoning_eating/imbalanced_phoning_female_case_2.pt')]
    path_test_dataset = Path('data/datasets/imSitu/data/phoning_eating/human_images/test') 
    is_cbm = False
    device = 'cuda'
    for path_model in path_models:
        main(path_model, is_cbm, path_test_dataset, device)
