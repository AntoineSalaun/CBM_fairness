import os
import random
from pathlib import Path
import json
import shutil

def gender_split(path_dataset):
    path_train_full = path_dataset / 'train/train_full'
    path_val = path_dataset / 'val'
    targets = os.listdir(path_val)
    targets.sort()


    path_train_male = path_dataset / 'gender_split' / 'male' / 'train'
    path_train_female = path_dataset / 'gender_split' / 'female' / 'train'
    path_val_male = path_dataset / 'gender_split' / 'male' / 'val'
    path_val_female = path_dataset / 'gender_split' / 'female' / 'val'

    for target in targets:
        path_train_male_target = path_train_male / target
        path_val_male_target = path_val_male / target
        
        path_train_female_target = path_train_female / target
        path_val_female_target = path_val_female / target

        if not os.path.exists(path_train_male_target):
            os.makedirs(path_train_male_target)
            os.makedirs(path_val_male_target)

        if not os.path.exists(path_train_female_target):
            os.makedirs(path_val_female_target)
            os.makedirs(path_train_female_target)

        files_val = os.listdir(path_val / target)
        for file in files_val:
            shutil.copy(path_val / target / file, path_val_male_target / file)
            shutil.copy(path_val / target / file, path_val_female_target / file)

        files_train_male = os.listdir(path_train_full / target / 'male')
        files_train_female = os.listdir(path_train_full / target / 'female')
        
        for file in files_train_male:
            shutil.copy(path_train_full / target / 'male' / file, path_train_male_target / file)
        for file in files_train_female:
            shutil.copy(path_train_full / target / 'female' / file, path_train_female_target / file)

       
if __name__ == "__main__":
    path_dataset = Path.cwd() / 'data/datasets/imSitu/data/200_verbs'
    gender_split(path_dataset)