from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

percent = 100

data_dir = '/root/projects/datasets/doctor_nurse/test/'
data_dir = 'data/datasets/doctor_nurse_2/balanced_train_test/test'

image_datasets = {'doctor_f': datasets.ImageFolder(data_dir + 'doctors/female',
                                          data_transforms['test']),
                  'nurse_f': datasets.ImageFolder(data_dir + 'nurses/female',
                                          data_transforms['test']),
                  'doctor_m': datasets.ImageFolder(data_dir + 'doctors/male',
                                          data_transforms['test']),
                  'nurse_m': datasets.ImageFolder(data_dir + 'nurses/male',
                                          data_transforms['test'])
                  }
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=False, num_workers=4)
               for x in ['doctor_f', 'doctor_m', 'nurse_f', 'nurse_m']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['doctor_f', 'doctor_m', 'nurse_f', 'nurse_m']}

model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, 2)
model.load_state_dict(torch.load('models/' + str(percent) + '.pth'))

model.eval()   # Set model to evaluate mode
df_corrects = 0
dm_corrects = 0
nf_corrects = 0
nm_corrects = 0

#Class 0 is doctor, class 1 is nurse
for inputs, labels in dataloaders['doctor_f']:
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if preds == 0:
        df_corrects += 1

print(df_corrects / dataset_sizes['doctor_f'])

for inputs, labels in dataloaders['doctor_m']:
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if preds == 0:
        dm_corrects += 1

print(dm_corrects / dataset_sizes['doctor_m'])

for inputs, labels in dataloaders['nurse_f']:
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if preds == 1:
        nf_corrects += 1

print(nf_corrects / dataset_sizes['nurse_f'])

for inputs, labels in dataloaders['nurse_m']:
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if preds == 1:
        nm_corrects += 1

print(nm_corrects / dataset_sizes['nurse_m'])

