import os
from pathlib import Path
import shutil
import random

def train_val_test_split(original_data_path,new_data_path, train_ratio, val_ratio=None, random_seed=0):
    """
    Splits the data from original_data_path into train, validation and test sets based on the provided ratios.
    If new_data_path is provided, the new datasets will be created in that directory. Otherwise, they will be created
    in the same directory as the original data.
    """
    random.seed(random_seed)

    # Get the list of class folders in the original data path
    class_folders = os.listdir(original_data_path)

    train_path = os.path.join(new_data_path, 'train')
    test_path = os.path.join(new_data_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    if val_ratio is not None:
        val_path = os.path.join(new_data_path, 'val')
        os.makedirs(val_path, exist_ok=True)

    # Split the data for each class folder
    for class_folder in class_folders:
        class_path = os.path.join(original_data_path, class_folder)
        images = os.listdir(class_path)
        random.shuffle(images)
        num_images = len(images)
        train_end = int(num_images * train_ratio)
        if val_ratio is not None:
            val_end = int(num_images * (train_ratio + val_ratio))
        else: 
            val_end = -1

        # Copy the images to the new directories
        for i, image in enumerate(images):
            src_path = os.path.join(class_path, image)
            if i < train_end:
                dst_path = os.path.join(train_path, class_folder, image)
            elif i < val_end:
                dst_path = os.path.join(val_path, class_folder, image)
            else:
                dst_path = os.path.join(test_path, class_folder, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

random.seed(0)

original_balanced_dataset = Path('data/datasets/imSitu/data/phoning_eating/human_images/datasets/original_balanced_halved')
original_balanced_tts = original_balanced_dataset.parent.parent / 'train_test_split' / 'gender_balanced'
train_val_test_split(original_balanced_dataset, original_balanced_tts, train_ratio=0.75)

imbalanced_binary_case_1_dataset = Path('data/datasets/imSitu/data/phoning_eating/human_images/datasets/imbalanced_binary_case_1')
imbalanced_binary_case_1_tts = imbalanced_binary_case_1_dataset.parent.parent / 'train_test_split' / 'imbalanced_binary_case_1'
train_val_test_split(imbalanced_binary_case_1_dataset, imbalanced_binary_case_1_tts, train_ratio=0.75)

imbalanced_binary_case_2_dataset = Path('data/datasets/imSitu/data/phoning_eating/human_images/datasets/imbalanced_binary_case_2')
imbalanced_binary_case_2_tts = imbalanced_binary_case_1_dataset.parent.parent / 'train_test_split' / 'imbalanced_binary_case_2'
train_val_test_split(imbalanced_binary_case_2_dataset, imbalanced_binary_case_2_tts, train_ratio=0.75)




