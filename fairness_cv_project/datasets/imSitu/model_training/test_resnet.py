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
from PIL import Image

def is_grayscale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

def find_grayscale_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if is_grayscale(os.path.join(directory, filename)):
                print(filename)

class SingleFolderDataset(Dataset):
    def __init__(self, img_dir, target_class, transform=None):
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

def load_resnet_model(path_resnet_model, num_classes, device):
    target_model = models.resnet50()
    target_model.fc = nn.Linear(target_model.fc.in_features, num_classes)

    state_dict = torch.load(path_resnet_model, map_location='cpu')
    target_model.load_state_dict(state_dict)
    target_model = target_model.to(device)

    return target_model

def read_txt(path_txt):
    with open(path_txt, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def main(path_model, is_cbm, path_verbs, path_test_dataset, device, num_classes=0):
    if is_cbm:
        target_model = load_cbm_model(path_model, device)
    else:
        target_model = load_resnet_model(path_model, num_classes, device)

    # Define your transform
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Added this line to handle images with alpha channel
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    categories = read_txt(path_verbs)
    genders = ['male', 'female']
    
    accuracies = {}
    target_model.eval()

    for category in categories:
        accuracies[category] = {}
        target_index = categories.index(category)

        for gender in genders:
            find_grayscale_images(path_test_dataset / category / gender)
            # Get the data
            folder_name = f"{category}/{gender}"
            dataset = SingleFolderDataset(f'{path_test_dataset}/{folder_name}', target_class=target_index, transform=data_transforms)  # or 1 for 'phoning'
            # Create the dataloader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

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

            # print(f'Accuracy of the network on {folder_name} test images: {100  * correct / total }')
            accuracies[category][gender] = {'accuracy': 100 * correct / total, 'correct': correct,'count': total}
    
    return accuracies

def main_genderless(path_model, is_cbm, path_verbs, path_test_dataset, path_val_dataset, device, num_classes=0):
    if is_cbm:
        target_model = load_cbm_model(path_model, device)
    else:
        target_model = load_resnet_model(path_model, num_classes, device)

    # Define your transform
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Added this line to handle images with alpha channel
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # categories = ['eating', 'phoning'] #put by alphabetical order
    categories = read_txt(path_verbs)
    
    accuracies = {}

    target_model.eval()
    for category in categories:
        accuracies[category] = {}
        target_index = categories.index(category)

        # Get the data
        folder_name = f"{category}"
        dataset = SingleFolderDataset(f'{path_test_dataset}/{folder_name}', target_class=target_index, transform=data_transforms)  # or 1 for 'phoning'
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

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

        # print(f'Accuracy of the network on {folder_name} test images: {100  * correct / total }')
        accuracies[category] = {'accuracy': 100 * correct / total, 'correct': correct,'count': total}
    
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



def compute_accuracies_by_group(accuracies, path_list_verbs):
    list_verbs = read_txt(path_list_verbs)
    list_verbs_1 = list_verbs[:len(list_verbs)//2]
    list_verbs_2 = list_verbs[len(list_verbs)//2:]

    results = {}

    count = {'male_1': 0, 'male_2': 0, 'female_1': 0, 'female_2': 0}
    correct = {'male_1': 0, 'male_2': 0, 'female_1': 0, 'female_2': 0} 

    for target in list_verbs:
        suffix = 1 if target in list_verbs_1 else 2
        for gender in ['male', 'female']:
            key = f'{gender}_{suffix}'
            count[key] += accuracies[target][gender]['count']
            correct[key] += accuracies[target][gender]['correct']

    for key in count:
        results[key] = {'accuracy': correct[key] / count[key], 
                        'correct': correct[key], 
                        'count': count[key]}
    
    results['male'] = {'accuracy': (results['male_1']['correct'] + results['male_2']['correct']) / (results['male_1']['count'] + results['male_2']['count']),
                        'correct': results['male_1']['correct'] + results['male_2']['correct'],
                        'count': results['male_1']['count'] + results['male_2']['count'] }

    results['female'] = {'accuracy': (results['female_1']['correct'] + results['female_2']['correct']) / (results['female_1']['count'] + results['female_2']['count']),
                         'correct': results['female_1']['correct'] + results['female_2']['correct'],
                         'count': results['female_1']['count'] + results['female_2']['count'] }


    results['Group 1'] = {'accuracy': (results['male_1']['correct'] + results['female_1']['correct']) / (results['male_1']['count'] + results['female_1']['count']),
                          'correct': results['male_1']['correct'] + results['female_1']['correct'],
                          'count': results['male_1']['count'] + results['female_1']['count']}
                        
    results['Group 2'] = {'accuracy': (results['male_2']['correct'] + results['female_2']['correct']) / (results['male_2']['count'] + results['female_2']['count']),
                          'correct': results['male_2']['correct'] + results['female_2']['correct'],
                          'count': results['male_2']['count'] + results['female_2']['count']}
    
    results['Total'] = {'accuracy': (results['Group 1']['correct'] + results['Group 2']['correct']) / (results['Group 1']['count'] + results['Group 2']['count']),
                        'correct': results['Group 1']['correct'] + results['Group 2']['correct'],
                        'count': results['Group 1']['count'] + results['Group 2']['count']}

    return results


def compute_parities_by_group(accuracies, results, path_list_verbs):
    list_verbs = read_txt(path_list_verbs)
    list_verbs_1 = list_verbs[:len(list_verbs)//2]
    list_verbs_2 = list_verbs[len(list_verbs)//2:]
    list_verbs_1_parities = dict()
    list_verbs_2_parities = dict()

    for target in list_verbs:
        suffix = 1 if target in list_verbs_1 else 2
        target_parity = (accuracies[target]['male']['correct'] / accuracies[target]['male']['count']) - (accuracies[target]['female']['correct'] / accuracies[target]['female']['count'])
        if suffix == 1:
            list_verbs_1_parities[target] = target_parity
        else:
            list_verbs_2_parities[target] = target_parity


    list_verbs_1_per_class_parity = 0
    for action in list_verbs_1_parities:
        list_verbs_1_per_class_parity += abs(list_verbs_1_parities[action])
    list_verbs_1_per_class_parity /= len(list_verbs_1_parities)

    list_verbs_2_per_class_parity = 0
    for action in list_verbs_2_parities:
        list_verbs_2_per_class_parity += abs(list_verbs_2_parities[action])
    list_verbs_2_per_class_parity /= len(list_verbs_2_parities)

    verbs_parities = list_verbs_1_parities | list_verbs_2_parities
    per_class_parity = 0
    for action in verbs_parities:
        per_class_parity += abs(verbs_parities[action])
    per_class_parity /= len(verbs_parities)
    male_top_three = sorted(verbs_parities.items(), key=lambda x: x[1], reverse=True)[:3]
    male_top_three_str = male_top_three[0][0] + " (" + str(male_top_three[0][1]) + "), " 
    male_top_three_str += male_top_three[1][0] + " (" + str(male_top_three[1][1]) + "), " 
    male_top_three_str += male_top_three[2][0] + " (" + str(male_top_three[2][1]) + ")"
    female_top_three = sorted(verbs_parities.items(), key=lambda x: x[1])[:3]
    female_top_three_str = female_top_three[0][0] + " (" + str(female_top_three[0][1]) + "), " 
    female_top_three_str += female_top_three[1][0] + " (" + str(female_top_three[1][1]) + "), " 
    female_top_three_str += female_top_three[2][0] + " (" + str(female_top_three[2][1]) + ")"


    results['Group 1']['Accuracy Parity'] = abs(results['male_1']['accuracy'] - results['female_1']['accuracy'])
    results['Group 1']['Accuracy Parity Class'] = 'male' if results['male_1']['accuracy'] > results['female_1']['accuracy'] else 'female'
    results['Group 1']['Per Class Accuracy Parity'] = list_verbs_1_per_class_parity

    results['Group 2']['Accuracy Parity'] = abs(results['male_2']['accuracy'] - results['female_2']['accuracy'])
    results['Group 2']['Accuracy Parity Class'] = 'male' if results['male_2']['accuracy'] > results['female_2']['accuracy'] else 'female'
    results['Group 2']['Per Class Accuracy Parity'] = list_verbs_2_per_class_parity

    results['Total']['Accuracy Parity'] = abs(results['male']['accuracy'] - results['female']['accuracy'])
    results['Total']['Accuracy Parity Class'] = 'male' if results['male']['accuracy'] > results['female']['accuracy'] else 'female'
    results['Total']['Per Class Accuracy Parity'] = per_class_parity
    results['Total']['Male Parity Top 3 Classes'] = male_top_three_str
    results['Total']['Female Parity Top 3 Classes'] = female_top_three_str

    return results
            

def compute_average_acc(accuracies, path_list_verbs, gender=True):
    list_verbs = read_txt(path_list_verbs)
    count_total = 0
    correct_total = 0
    for target in list_verbs:
        if gender:
            for gender in ['male', 'female']:
                correct_total += accuracies[target][gender]['correct']
                count_total += accuracies[target][gender]['count']
        else:
            correct_total += accuracies[target]['correct']
            count_total += accuracies[target]['count']
    
    print(correct_total / count_total)
    return correct_total / count_total


def count_male_dominant_classes(accuracies, list_verbs):

    count = 0
    for target in list_verbs:
        if accuracies[target]['male']['accuracy'] > accuracies[target]['female']['accuracy']:
            count += 1

    return count, count / len(list_verbs)


def get_count_males_result(accuracies, accuracy_by_group, path_list_verbs):

    list_verbs = read_txt(path_list_verbs)
    list_verbs_1 = list_verbs[:len(list_verbs)//2]
    list_verbs_2 = list_verbs[len(list_verbs)//2:]

    count_1, proportion_1 = count_male_dominant_classes(accuracies, list_verbs_1)
    count_2, proportion_2 = count_male_dominant_classes(accuracies, list_verbs_2)

    accuracy_parity = abs(accuracy_by_group['male']['accuracy'] - accuracy_by_group['female']['accuracy'])
    accuracy_parity_class = 'male' if (accuracy_by_group['male']['accuracy'] - accuracy_by_group['female']['accuracy']) > 0 else 'female'
    per_class_accuracy_parity = accuracy_by_group['Total']['Per Class Accuracy Parity']

    results = {'Count male 1': count_1, 
               'Proportion 1': proportion_1, 
               'Count male 2': count_2, 
               'Proportion 2': proportion_2, 
               'Count total': count_1 + count_2, 
               'Proportion total': (count_1 + count_2) / len(list_verbs), 
               'Length Total': len(list_verbs), 
               'accuracy_parity': accuracy_parity, 
               'accuracy_parity_class': accuracy_parity_class,
               'per_class_accuracy_parity': per_class_accuracy_parity,
               'Male Parity Top 3 Classes': accuracy_by_group['Total']['Male Parity Top 3 Classes'],
               'Female Parity Top 3 Classes': accuracy_by_group['Total']['Female Parity Top 3 Classes'],
               }

    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_cbm', action='store_true', default=False)
    parser.add_argument('--path_model', type=str, default='saved_models/imSitu/30_verbs/baseline/balanced/model.pt')
    parser.add_argument('--path_test_dataset', type=str, default='data/datasets/imSitu/data/30_verbs/test_with_gender')
    parser.add_argument('--path_list_verbs', type=str, default='data/datasets/imSitu/data/30_verbs/verbs.txt')
    parser.add_argument('--path_result', type=str, default='results/imSitu/30_verbs/baseline/balanced')
    parser.add_argument('--num_classes', type=int, default=30)

    args = parser.parse_args()
    path_model = Path(args.path_model)
    path_test_dataset = Path(args.path_test_dataset)
    path_list_verbs = Path(args.path_list_verbs)
    num_classes = args.num_classes
    path_result = Path(args.path_result)
    is_cbm = args.is_cbm

    device = 'cuda'

    accuracies_main = main(path_model, is_cbm, path_list_verbs, path_test_dataset, device, num_classes)
    average_acc = compute_average_acc(accuracies_main, path_list_verbs, True)
    accuracies_by_group = compute_accuracies_by_group(accuracies_main, path_list_verbs)
    accuracies_by_group = compute_parities_by_group(accuracies_main, accuracies_by_group, path_list_verbs)
    count_male = get_count_males_result(accuracies_main, accuracies_by_group, path_list_verbs)

    if path_result is not None:
        df = pd.DataFrame(accuracies_by_group).transpose()

        if not path_result.exists():
            path_result.mkdir(parents=True)
            
        df.to_csv(path_result / 'accuracies_by_group.csv')
        with open(path_result / 'accuracy.txt', 'w') as f:
            f.write(str(average_acc))
            f.write('\n')
            f.write(str(count_male))