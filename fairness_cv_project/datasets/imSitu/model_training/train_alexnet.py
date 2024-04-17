from __future__ import print_function, division

import argparse 
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path 
import random 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import warnings
warnings.filterwarnings("ignore")

def train(path_dataset: Path, path_save: Path, args: argparse.Namespace):

    LR = args.lr
    MOMENTUM = args.momentum
    STEP_SIZE = args.step_size
    GAMMA = args.gamma
    NUM_EPOCHS = args.num_epochs

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
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
     
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                        shuffle=True, num_workers=15)
                for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    num_classes = len(image_datasets['train'].classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) 
       
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
            
            if phase == 'train':
                scheduler.step()

                
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_acc

    model_conv = models.alexnet()
    model_conv.load_state_dict(torch.load(Path.cwd() / 'saved_models/alexnet.pt'))
                        
    for param in model_conv.features.parameters():
        param.requires_grad = False

    model_conv.classifier[6] = nn.Linear(4096, num_classes)
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=LR, momentum=MOMENTUM)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=STEP_SIZE, gamma=GAMMA)

    model_conv, best_acc = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=NUM_EPOCHS)

    
    folders = str(path_dataset).split('/')
    name_dataset = '-'.join(folders[-2:])

    if not os.path.exists(path_save.parent):
        os.makedirs(path_save.parent)
    torch.save(model_conv.state_dict(), path_save)

    with open(path_save.parent / 'results.txt', 'a') as f:
        f.write(f'{name_dataset}: {best_acc}\n')

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    LR = 0.01
    MOMENTUM = 0.9
    STEP_SIZE = 7
    GAMMA = 0.1
    NUM_EPOCHS = 10
    SEED = 0

    parser = argparse.ArgumentParser(description='Settings for training')
    parser.add_argument("--dataset", type=str, default='200_verbs/train_test_split/train_balanced', help="dataset name")
    parser.add_argument("--save_name", type=str, default='200_verbs/baseline/balanced.pt', help="name of the saved model")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="momentum")
    parser.add_argument("--step_size", type=int, default=STEP_SIZE, help="step size")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="gamma")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="number of epochs")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")

    random.seed(SEED)
    torch.manual_seed(SEED)

    args = parser.parse_args()

    path_root = Path.cwd() / 'data' / 'datasets' / 'imSitu' 
    path_dataset = path_root / 'data' / args.dataset
    path_save = Path(f'saved_models/imSitu/{args.save_name}')

    print(f'Training on {args.dataset}')
    train(path_dataset, path_save, args)