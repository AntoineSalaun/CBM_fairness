import sys
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))

import os
import torch
from pathlib import Path
import torch.nn as nn
from torchvision import datasets, transforms, models

from fairness_cv_project.methods.label_free_cbm.src.clip import clip
from pytorchcv.model_provider import get_model as ptcv_get_model

# This was a magic path, now it should be Path.cwd() as the working path should always be the root path
# If it doesn't work, just hard code it
PATH_ROOT = Path.cwd()

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/datasets/CUB/train",
    "cub_val":"data/datasets/CUB/test",

    "doctor_nurse_full_train": "data/datasets/doctor_nurse_full/train", 
    "doctor_nurse_full_val": "data/datasets/doctor_nurse_full/val",
    
    "doctor_nurse_gender_biased_train": "data/datasets/doctor_nurse_gender_biased/train", 
    "doctor_nurse_gender_biased_val": "data/datasets/doctor_nurse_gender_biased/val",

    "doctor_nurse_balanced_train": 'data/datasets/doctor_nurse_2/train_test_split/train_balanced/train',
    "doctor_nurse_balanced_val": 'data/datasets/doctor_nurse_2/train_test_split/train_balanced/test',
    
    "doctor_nurse_imbalanced_1_train": 'data/datasets/doctor_nurse_2/train_test_split/train_imbalanced_1/train',
    "doctor_nurse_imbalanced_1_val": 'data/datasets/doctor_nurse_2/train_test_split/train_imbalanced_1/test', 

    "doctor_nurse_imbalanced_2_train": 'data/datasets/doctor_nurse_2/train_test_split/train_imbalanced_2/train',
    "doctor_nurse_imbalanced_2_val": 'data/datasets/doctor_nurse_2/train_test_split/train_imbalanced_2/test',

    'phoning_eating_balanced_train': 'data/datasets/imSitu/data/phoning_eating/human_images/train_test_split/balanced/train',
    'phoning_eating_balanced_val': 'data/datasets/imSitu/data/phoning_eating/human_images/train_test_split/balanced/test',
    'phoning_eating_imbalanced_1_train': 'data/datasets/imSitu/data/phoning_eating/human_images/train_test_split/imbalanced_1/train',
    'phoning_eating_imbalanced_1_val': 'data/datasets/imSitu/data/phoning_eating/human_images/train_test_split/imbalanced_1/test',
    'phoning_eating_imbalanced_2_train': 'data/datasets/imSitu/data/phoning_eating/human_images/train_test_split/imbalanced_2/train',
    'phoning_eating_imbalanced_2_val': 'data/datasets/imSitu/data/phoning_eating/human_images/train_test_split/imbalanced_2/test',

    'imSitu_30_balanced_train': 'data/datasets/imSitu/data/30_verbs/train_val_split/train_balanced/train',
    'imSitu_30_balanced_val': 'data/datasets/imSitu/data/30_verbs/train_val_split/train_balanced/val',
    
    'imSitu_30_imbalanced_1_train': 'data/datasets/imSitu/data/30_verbs/train_val_split/train_imbalanced_1/train',
    'imSitu_30_imbalanced_1_val': 'data/datasets/imSitu/data/30_verbs/train_val_split/train_imbalanced_1/val',

    'imSitu_30_imbalanced_2_train': 'data/datasets/imSitu/data/30_verbs/train_val_split/train_imbalanced_2/train',
    'imSitu_30_imbalanced_2_val': 'data/datasets/imSitu/data/30_verbs/train_val_split/train_imbalanced_2/val',

    'imSitu_30_full_train': 'data/datasets/imSitu/data/30_verbs_full/train_val/train',
    'imSitu_30_full_val' : 'data/datasets/imSitu/data/30_verbs_full/train_val/val',

    'imSitu_200_full_train': 'data/datasets/imSitu/data/200_verbs_full/train_val/train',
    'imSitu_200_full_val' : 'data/datasets/imSitu/data/200_verbs_full/train_val/val',

    'imSitu_200_balanced_train': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_balanced/train',
    'imSitu_200_balanced_val': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_balanced/val',

    'imSitu_200_imbalanced_1_train': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_imbalanced_1/train',
    'imSitu_200_imbalanced_1_val': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_imbalanced_1/val',

    'imSitu_200_imbalanced_2_train': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_imbalanced_2/train',
    'imSitu_200_imbalanced_2_val': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_imbalanced_2/val',

    'imSitu_200_male_train': 'data/datasets/imSitu/data/200_verbs/gender_split/male/train',
    'imSitu_200_male_val': 'data/datasets/imSitu/data/200_verbs/gender_split/male/val',

    'imSitu_200_female_train': 'data/datasets/imSitu/data/200_verbs/gender_split/female/train',
    'imSitu_200_female_val': 'data/datasets/imSitu/data/200_verbs/gender_split/female/val',

    'imSitu_200_shrunk_train': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_shrunk/train',
    'imSitu_200_shrunk_val': 'data/datasets/imSitu/data/200_verbs/train_val_split/train_shrunk/val',

    'imSitu_200_daniel_imbalanced_train': 'data/datasets/imSitu/data/200_verbs_daniel/200_verbs_imbalanced/train',
    'imSitu_200_daniel_imbalanced_val': 'data/datasets/imSitu/data/200_verbs_daniel/200_verbs_imbalanced/val',

    'imSitu_200_daniel_balanced_train': 'data/datasets/imSitu/data/200_verbs_daniel/200_verbs_balanced/train',
    'imSitu_200_daniel_balanced_val': 'data/datasets/imSitu/data/200_verbs_daniel/200_verbs_balanced/val',
}

LABEL_FILES = {"places365":"data/classes/categories_places365_clean.txt",
               "imagenet":"data/classes/imagenet_classes.txt",
               "cifar10":"data/classes/cifar10_classes.txt",
               "cifar100":"data/classes/cifar100_classes.txt",
               "cub":"data/classes/cub_classes.txt",

               "doctor_nurse_full":"data/classes/doctor_nurse_classes.txt",
               "doctor_nurse_gender_biased": "data/classes/doctor_nurse_classes.txt",
               'doctor_nurse_balanced': "data/classes/doctor_nurse_classes.txt",
               'doctor_nurse_imbalanced_1': "data/classes/doctor_nurse_classes.txt",
               'doctor_nurse_imbalanced_2': "data/classes/doctor_nurse_classes.txt",

               'phoning_eating_balanced': 'data/classes/phoning_eating_classes.txt',
               'phoning_eating_imbalanced_1': 'data/classes/phoning_eating_classes.txt',
               'phoning_eating_imbalanced_2': 'data/classes/phoning_eating_classes.txt',

               'imSitu_30': 'data/classes/imSitu_30_classes.txt',
               'imSitu_30_balanced': 'data/classes/imSitu_30_classes.txt',
               'imSitu_30_imbalanced_1': 'data/classes/imSitu_30_classes.txt',
               'imSitu_30_imbalanced_2': 'data/classes/imSitu_30_classes.txt',
               'imSitu_30_full': 'data/classes/imSitu_30_classes.txt',

               'imSitu_200': 'data/classes/imSitu_200_classes.txt',
               'imSitu_200_full': 'data/classes/imSitu_200_classes.txt',

               'imSitu_200_balanced': 'data/classes/imSitu_200_classes.txt',
               'imSitu_200_imbalanced_1': 'data/classes/imSitu_200_classes.txt',
               'imSitu_200_imbalanced_2': 'data/classes/imSitu_200_classes.txt',

               'imSitu_200_male': 'data/classes/imSitu_200_classes.txt',
               'imSitu_200_female': 'data/classes/imSitu_200_classes.txt',

               'imSitu_200_shrunk': 'data/classes/imSitu_200_classes.txt',
               'imSitu_200_daniel_imbalanced': 'data/classes/imSitu_200_classes.txt',
               'imSitu_200_daniel_balanced': 'data/classes/imSitu_200_classes.txt',
}


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_alexnet_doctor_nurse_preprocess():
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert('RGBA')),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img[:3, :, :]), # Drop alpha channel
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Lambda(lambda img: img.convert('RGBA')),
        ]) 
    
    return preprocess



def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data_path = Path.cwd() / 'data' / 'datasets' / 'cifar100'
        data = datasets.CIFAR100(root=data_path, train=True, download=False, transform=preprocess)

    elif dataset_name == "cifar100_val":
        data_path = Path.cwd() / 'data' / 'datasets' / 'cifar100'
        data = datasets.CIFAR100(root=data_path, train=False, download=False, transform=preprocess)

    elif dataset_name == "cifar10_train":
        data_path = Path.cwd() / 'data' / 'datasets' / 'cifar10'
        data = datasets.CIFAR10(root=data_path, train=True, download=False, transform=preprocess)

    elif dataset_name == "cifar10_val": 
        data_path = Path.cwd() / 'data' / 'datasets' / 'cifar10'
        data = datasets.CIFAR10(root=data_path, train=False, download=False, transform=preprocess)

    elif dataset_name == "places365_train":
        data_path = Path.cwd() / 'data' / 'datasets' / 'places365'
        data = datasets.Places365(root=data_path, split="train-standard", small=True, download=False, transform=preprocess)
            
    elif dataset_name == "places365_val": 
        data_path = Path.cwd() / 'data' / 'datasets' / 'places365'
        data = datasets.Places365(root=data_path, split="val", small=True, download=False, transform=preprocess)
    
    elif dataset_name in DATASET_ROOTS.keys():
        if 'doctor_nurse' in dataset_name:
            preprocess = get_alexnet_doctor_nurse_preprocess()
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    
    else:
        raise ValueError("Dataset name not found")
    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets

def get_target_model(target_name, device, d_probe='', path_root=PATH_ROOT):

    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name.startswith('alexnet_doctor_nurse'):
        
        # target_model = models.alexnet()
        # target_model.classifier[6] = nn.Linear(4096, 2)
        # state_dict = torch.load(path_root / f'saved_models/doctor_nurse_alexnet/{target_name}.pt', map_location='cpu')
        # target_model = target_model.load_state_dict(state_dict)

        target_model = models.alexnet()
        target_model.classifier[6] = nn.Linear(4096, 2)
        path_alexnet_model = Path.cwd() / 'saved_models' / 'alexnet.pt'
        # state_dict = torch.load(path_alexnet_model, map_location='cpu')
        # target_model.load_state_dict(state_dict)
        target_model = target_model.to(device)

        target_model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert('RGBA')),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img[:3, :, :]), # Drop alpha channel
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) 

    elif target_name.startswith('alexnet_phoning_eating'):
        target_model = models.alexnet()

        path_alexnet_model = Path.cwd() / 'saved_models' / 'alexnet.pt'
        state_dict = torch.load(path_alexnet_model, map_location='cpu')
        target_model.load_state_dict(state_dict)
        # target_model.classifier[6] = nn.Linear(4096, 2)
        target_model = target_model.to(device)

        target_model.eval()

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

        if d_probe.endswith('train'):
            preprocess = data_transforms['train']
        elif d_probe.endswith('val'):
            preprocess = data_transforms['test']
        else:
            preprocess = None


    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        path = path_root / 'saved_models' / 'resnet18_cub.pt'
        target_model = torch.load(path).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    return target_model, preprocess