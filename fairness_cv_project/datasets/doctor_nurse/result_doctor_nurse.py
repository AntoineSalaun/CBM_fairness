import argparse
import os
import json
from pathlib import Path
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, str(Path.cwd()))

from fairness_cv_project.methods.label_free_cbm.src.utils import data_utils
from fairness_cv_project.methods.label_free_cbm.src.models import cbm
from fairness_cv_project.methods.label_free_cbm.src.plots import plots

# Load the model
def load_cbm_model(load_dir, device):
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
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

def create_dataloaders(path_dataset):
    # Create the different dataloaders
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    # Create one dataloader with the 4 classes dr_f dr_m nur_f nur_m
    dataloader_full = datasets.ImageFolder(path_dataset, data_transforms)

    # Initialize an empty dictionary to hold 4 data loaders, each one having one class
    dataloaders = {}

    # For each class in your data...
    for i in range(len(dataloader_full.classes)):
        # Create a subset of your data that only includes images of this class
        subset = torch.utils.data.Subset(
            dataloader_full,
            [j for j, target in enumerate(dataloader_full.targets) if target == i]
        )

        # Create a data loader for this subset of data
        dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

        # Add the data loader to your dictionary of data loaders
        dataloaders[dataloader_full.classes[i]] = dataloader
    
    return dataloaders

def compute_accuracies(model, dataloaders, device):
    def get_exact_num_samples(dataloader):
        num_samples = 0
        for data in dataloader:
            num_samples += data[0].shape[0]
        return num_samples

    def compute_accuracy(model, dataloader, doctor):
        target_class = 0 if doctor else 1
        correct_preds = 0
        #Class 0 is doctor, class 1 is nurse
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, 1)
            correct_preds += torch.sum(preds == target_class).item()

        return correct_preds / get_exact_num_samples(dataloader)

    accuracies = {}
    for key, dataloader in dataloaders.items():
        is_doctor = key.startswith('doctors')
        accuracies[key] = compute_accuracy(model, dataloader, is_doctor)
    return accuracies

def compute_fairness_metrics(accuracies):
    accuracies['doctors_male_minus_female'] = accuracies['doctors_male'] - accuracies['doctors_female']
    accuracies['nurses_male_minus_female'] = accuracies['nurses_male'] - accuracies['nurses_female']
    return accuracies

def plot_accuracies(accuracies, path_result):
    # Create a 2x2 grid of subplots
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    # Set the title of the plot
    fig.suptitle('Accuracy by Gender and Profession')

    # Set the labels for the x and y axes
    ax[0, 0].set_ylabel('Doctor')
    ax[1, 0].set_ylabel('Nurse')
    ax[1, 0].set_xlabel('Male')
    ax[1, 1].set_xlabel('Female')

    # Set the values for each cell in the grid
    ax[0, 0].text(0.5, 0.5, f'{accuracies["doctors_male"]:.2f}', ha='center', va='center', fontsize=20)
    ax[0, 1].text(0.5, 0.5, f'{accuracies["doctors_female"]:.2f}', ha='center', va='center', fontsize=20)
    ax[1, 0].text(0.5, 0.5, f'{accuracies["nurses_male"]:.2f}', ha='center', va='center', fontsize=20)
    ax[1, 1].text(0.5, 0.5, f'{accuracies["nurses_female"]:.2f}', ha='center', va='center', fontsize=20)

    # Remove the ticks from the x and y axes
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    # Display the plot
    plt.show()
    return plt 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, help='path to the trained model')
    parser.add_argument('--path_alexnet_model', type=str, help='path to the alexnet model')
    parser.add_argument('--path_dataset', type=str, default='data/datasets/doctor_nurse_stratified/full', help='path to the dataset')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training and inference')
    
    args = parser.parse_args()

    device = args.device
    path_dataset = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse_2' / 'balanced_train_test' / 'test'

    if args.load_dir:
        load_dir = Path(f"{args.load_dir}")
        path_result = Path.cwd() / 'results' / 'doctor_nurse' / f"results_{args.load_dir.split('/')[-1]}" 
        model = load_cbm_model(load_dir, device)
    else:
        path_alexnet_model = Path.cwd() / 'saved_models' / 'alexnet.pt'
        path_result = Path.cwd() / 'results' / 'doctor_nurse' / f"results_{args.path_alexnet_model.split('/')[-1]}"
        model = load_alexnet_model(path_alexnet_model, device)
    
    print(path_result)
    path_result.mkdir(parents=True, exist_ok=True)

    dataloaders = create_dataloaders(path_dataset)

    accuracies = compute_accuracies(model, dataloaders, device)
    accuracies = compute_fairness_metrics(accuracies)
    with open(path_result / "accuracies", 'w') as f:
        json.dump(accuracies, f)

    plot = plot_accuracies(accuracies, path_result)

    plot.savefig(path_result / "plot.png")
