from __future__ import print_function, division

import argparse 
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import warnings
warnings.filterwarnings("ignore")

def train(path_dataset: Path):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {'train': datasets.ImageFolder(path_dataset / 'train',
                                            data_transforms['train']),
                    'test': datasets.ImageFolder(path_dataset / 'test',
                                            data_transforms['test'])
                    }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and testing phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to test mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                    # statistics 
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_acc

    model_conv = models.alexnet()
    model_conv.load_state_dict(torch.load(Path.cwd() / 'saved_models/alexnet.pt'))
                        
    for param in model_conv.features.parameters():
        param.requires_grad = False

    model_conv.classifier[6] = nn.Linear(4096, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv, best_acc = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=25)

    print(model_conv)
    folders = str(path_dataset).split('/')
    name_dataset = '-'.join(folders[-2:])
    torch.save(model_conv.state_dict(), Path.cwd() / 'saved_models/doctor_nurse_alexnet/alexnet_doctor_nurse' + name_dataset + '.pt')

    with open(Path.cwd() / 'saved_models/doctor_nurse_alexnet/results.txt', 'a') as f:
        f.write(f'{name_dataset}: {best_acc}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings for training')
    parser.add_argument("--dataset", type=str, default='default_train_test_75', help="dataset name")

    args = parser.parse_args()
    dataset = args.dataset

    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse'
    path_dataset = path_root / dataset
    train(path_dataset)