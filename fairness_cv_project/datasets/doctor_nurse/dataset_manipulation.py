import shutil
import random
import math
import os
from pathlib import Path



def remove_ash_extensions_image():
    """
    Remove all image files with the '.ash' extension from the 'data/datasets' directory and its subdirectories.

    This function uses the pathlib module to locate all files with the '.ash' extension in the 'data/datasets' directory and its subdirectories.
    It then deletes all the files with the '.ash' extension using the unlink() method of the Path object.

    The scrapper from the original dataset scrapped some files with an .ash extension, which are not displayable images.

    Returns:
        None 
    """
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse_2' 

    for path in path_root.glob('**/*.ash'):
        print(f'Deleting {path}')
        path.unlink()

    for path in path_root.glob('**/*purepng'):
        print(f'Deleting {path}')
        path.unlink()


def transfer_original_to_gender_dataset():
    """
    Transfer original dataset to gender dataset.

    The function copies the original dataset to a new directory structure that separates the images by gender and profession.
    The original dataset is located in the 'original_dataset' directory under the 'doctor_nurse' dataset directory.
    The new gender dataset is located in the 'original_dataset_gender' directory under the 'doctor_nurse' dataset directory.
    The new directory structure separates the images by gender and profession, with 'doctors' and 'nurses' directories for each gender.
    Each 'doctors' and 'nurses' directory contains 'male' and 'female' directories respectively.
    The function uses the shutil module to copy the images from the original dataset to the new gender dataset.
    The function also checks that the number of images in each directory is correct.

    Returns:
        None
    """
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse_2'
    path_original = path_root / 'original_dataset'
    path_gender_dataset = path_root / 'original_dataset_gender'

    path_dr_dark_male = path_original / 'dr' / 'mal_dr_dark_62'
    path_dr_light_male = path_original / 'dr' / 'mal_dr_light_308'
    path_dr_male = path_gender_dataset / 'doctors' / 'male'

    path_dr_dark_female = path_original / 'dr' / 'fem_dr_dark_56'
    path_dr_light_female = path_original / 'dr' / 'fem_dr_light_256'
    path_dr_female = path_gender_dataset / 'doctors' / 'female'

    path_nurses_dark_male = path_original / 'nurse' / 'mal_nurse_dark_76'
    path_nurses_light_male = path_original / 'nurse' / 'mal_nurse_light_203'
    path_nurses_male = path_gender_dataset / 'nurses' / 'male'

    path_nurses_dark_female = path_original / 'nurse' / 'fem_nurse_dark_63'
    path_nurses_light_female = path_original / 'nurse' / 'fem_nurse_light_252'
    path_nurses_female = path_gender_dataset / 'nurses' / 'female'

    shutil.copytree(path_dr_dark_male, path_dr_male, dirs_exist_ok=True)
    shutil.copytree(path_dr_light_male, path_dr_male, dirs_exist_ok=True)

    shutil.copytree(path_dr_dark_female, path_dr_female, dirs_exist_ok=True)
    shutil.copytree(path_dr_light_female, path_dr_female, dirs_exist_ok=True)

    shutil.copytree(path_nurses_dark_male, path_nurses_male, dirs_exist_ok=True)
    shutil.copytree(path_nurses_light_male, path_nurses_male, dirs_exist_ok=True)

    shutil.copytree(path_nurses_dark_female, path_nurses_female, dirs_exist_ok=True)
    shutil.copytree(path_nurses_light_female, path_nurses_female, dirs_exist_ok=True)


def count_files_dataset(path, verbose=1):
    """
    Count the number of files for dr male, dr female, nurses male, nurses females, printing them, and printing the percentage of male doctors compared to all doctors, male nurses compared to all nurses.

    The function uses the pathlib module to locate all files in the 'original_dataset_gender' directory under the 'doctor_nurse' dataset directory.
    It then counts the number of files for dr male, dr female, nurses male, nurses females, printing them, and printing the percentage of male doctors compared to all doctors, male nurses compared to all nurses.

    Returns:
        None 
    """
    path_dr_male = path / 'doctors' / 'male'
    path_dr_female = path / 'doctors' / 'female'
    path_nurses_male = path / 'nurses' / 'male'
    path_nurses_female = path / 'nurses' / 'female'

    num_dr_male = len(list(path_dr_male.glob('*.*')))
    num_dr_female = len(list(path_dr_female.glob('*.*')))
    num_nurses_male = len(list(path_nurses_male.glob('*.*')))
    num_nurses_female = len(list(path_nurses_female.glob('*.*')))

    total_doctors = num_dr_male + num_dr_female
    total_nurses = num_nurses_male + num_nurses_female
    
    if verbose> 0:
        print(f"Number of files for dr male: {num_dr_male}")
        print(f"Number of files for dr female: {num_dr_female}")
        print(f"Number of files for nurses male: {num_nurses_male}")
        print(f"Number of files for nurses female: {num_nurses_female}")

        print(f"Percentage of male doctors compared to all doctors: {num_dr_male/total_doctors*100:.2f}%")
        print(f"Percentage of male nurses compared to all nurses: {num_nurses_male/total_nurses*100:.2f}%")
        print(f"Percentage of female doctors compared to all doctors: {num_dr_female/total_doctors*100:.2f}%")
        print(f"Percentage of female nurses compared to all nurses: {num_nurses_female/total_nurses*100:.2f}%")

    return num_dr_male, num_dr_female, num_nurses_male, num_nurses_female


def count_files_train_test_set(path, print_files=False):
    """
    Count the number of files for dr, nurses, printing them, and printing the percentage of male doctors compared to all doctors, male nurses compared to all nurses.

    The function uses the pathlib module to locate all files in the 'default_train_test_75' directory under the 'doctor_nurse' dataset directory.
    It then counts the number of files for dr, nurses, printing them, and printing the percentage of male doctors compared to all doctors, male nurses compared to all nurses.

    Returns:
        None 
    """

    dr_train, dr_test = 0, 0
    nurses_train, nurses_test = 0, 0

    for dataset in ['train', 'test']:
        path_dr = path / dataset / 'doctors'
        path_nurses = path / dataset / 'nurses'
        
        num_dr_male = len(list(path_dr.glob('*_male.*')))
        num_dr_female = len(list(path_dr.glob('*_female.*')))
        num_nurses_male = len(list(path_nurses.glob('*_male.*')))
        num_nurses_female = len(list(path_nurses.glob('*_female.*')))

        total_doctors = num_dr_male + num_dr_female
        total_nurses = num_nurses_male + num_nurses_female

        if print_files:
            print(f"________DATASET = {dataset}________")
            print(f"Number of files for dr: {num_dr_male + num_dr_female}")
            print(f"Number of files for nurses: {num_nurses_male + num_nurses_female}")

            print(f"Number of files for dr male: {num_dr_male}")
            print(f"Number of files for dr female: {num_dr_female}")
            print(f"Number of files for nurses male: {num_nurses_male}")
            print(f"Number of files for nurses female: {num_nurses_female}")

            print(f"Percentage of male doctors compared to all doctors: {num_dr_male/total_doctors*100:.2f}%")
            print(f"Percentage of male nurses compared to all nurses: {num_nurses_male/total_nurses*100:.2f}%")
            print(f"Percentage of female doctors compared to all doctors: {num_dr_female/total_doctors*100:.2f}%")
            print(f"Percentage of female nurses compared to all nurses: {num_nurses_female/total_nurses*100:.2f}%")
        
        if dataset == 'train':
            dr_train = total_doctors
            nurses_train = total_nurses
        else:
            dr_test = total_doctors
            nurses_test = total_nurses
    
    print(f'Number of files for dr train: {dr_train}')
    print(f'Number of files for nurses train: {nurses_train}')
    print(f'Number of files for dr test: {dr_test}')
    print(f'Number of files for nurses test: {nurses_test}')

    print(f'Percentage of dr train compared to all doctors: {dr_train/(dr_train+dr_test)*100:.2f}%')
    print(f'Percentage of nurses train compared to all nurses: {nurses_train/(nurses_train+nurses_test)*100:.2f}%')

    return dr_train, nurses_train, dr_test, nurses_test

def create_balanced_dataset(path_root, num_samples: int = 262, random_state: int = 42):
    """
    Randomly selects 262 samples from each of the male nurses, male doctors, female nurses, and female doctors folders in the original_dataset_gender directory, and puts them in a new dataset in data/datasets/doctor_nurse/balanced_dataset.

    The file structure is the same as in data/datasets/doctor_nurse/original_dataset_gender, with a first folder for doctor / nurses and a second subfolder for male / female.
    
    Args:
            num_samples (int): The number of samples to select from each category/gender. Default is 262, number of nurse male.
    
    Returns:
        None
    """
    random.seed(random_state)
    path_gender_dataset = path_root / 'original_dataset_gender'
    path_balanced_dataset = path_root / 'balanced_dataset'

    # Create the balanced dataset directory if it doesn't exist
    if not path_balanced_dataset.exists():
        path_balanced_dataset.mkdir(parents=True)

    # Loop through each category (doctor/nurse, male/female)
    for category in ['doctors', 'nurses']:
        for gender in ['male', 'female']:
            # Create the subdirectory in the balanced dataset directory
            path_category_gender = path_balanced_dataset / category / gender
            if not path_category_gender.exists():
                path_category_gender.mkdir(parents=True)

            # Get a list of all files in the original dataset directory for this category/gender
            path_category_gender_original = path_gender_dataset / category / gender
            files = list(path_category_gender_original.glob('*.*'))

            # Randomly select 262 files
            selected_files = random.sample(files, num_samples)

            # Copy the selected files to the balanced dataset directory
            for file in selected_files:
                shutil.copy2(file, path_category_gender / file.name)


def create_balanced_train_test_set(ratio_train: float = 0.75, path_root: Path = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse'):
    """
    Creates a training set and a testing set containing doctors and nurses from `doctor_nurse/balanced_dataset`. The set should be in the folder `doctor_nurse/default_set_{suffix}`, and there should be two subfolders: `train` and `test`. Each subfolder should have two subfolders, `doctors` that contains male and female doctors, and `nurses` that contain male and female nurses. The ratio of training data compared the the whole dataset should come from the argument `ratio_train`.

    Args:
        ratio_train (float): The ratio of training data compared to the whole dataset. Default is 0.75.
        path_root (Path): The root path of the dataset. Default is Path.cwd() / 'data' / 'datasets' / 'doctor_nurse'.
        suffix (str): The suffix to add to the folder name. Default is an empty string.

    Returns:
        None
    """
    # Create the default set directory if it doesn't exist
    path_default_set = path_root / f'balanced_train_test'
    if not path_default_set.exists():
        path_default_set.mkdir(parents=True)

    # Loop through each category (doctor/nurse) and gender (male/female)
    for category in ['doctors', 'nurses']:
        for gender in ['male', 'female']:
            # Create the subdirectories in the default set directory
            path_train_category = path_default_set / 'train' / category
            path_test_category = path_default_set / 'test' / category
            if not path_train_category.exists():
                (path_train_category / 'male').mkdir(parents=True)
                (path_train_category / 'female').mkdir(parents=True)
            if not path_test_category.exists():
                (path_test_category / 'male').mkdir(parents=True)
                (path_test_category / 'female').mkdir(parents=True)

            # Get a list of all files in the balanced dataset directory for this category/gender
            path_category_gender_balanced = path_root / 'balanced_dataset' / category / gender
            files = list(path_category_gender_balanced.glob('*.*'))

            # Shuffle the files
            random.shuffle(files)

            # Split the files into training and testing sets
            num_train = int(len(files) * ratio_train)
            train_files = files[:num_train]
            test_files = files[num_train:]

            # Copy the training files to the default set directory
            for file in train_files:
                if gender == 'female':
                    new_filename = file.stem + '_female' + file.suffix
                else:
                    new_filename = file.stem + '_male' + file.suffix
                shutil.copy2(file, path_train_category / gender / new_filename)

            # Copy the testing files to the default set directory
            for file in test_files:
                if gender == 'female':
                    new_filename = file.stem + '_female' + file.suffix
                else:
                    new_filename = file.stem + '_male' + file.suffix
                shutil.copy2(file, path_test_category / gender / new_filename)


def create_train_dataset(path_dataset=Path.cwd() / 'data' / 'datasets' / 'doctor_nurse_2'):
    """
    We do the same as imSitu but in doctor nurse 
    """
    path_train_full = path_dataset / 'balanced_train_test' / 'train'
    
    path_train = path_dataset / 'train_test_split'
        
    path_train_balanced = path_train / 'train_balanced' / 'train'
    path_train_imbalanced_1 = path_train / 'train_imbalanced_1' / 'train'
    path_train_imbalanced_2 = path_train / 'train_imbalanced_2' / 'train'


    if not path_train_balanced.exists():
        (path_train_imbalanced_1).mkdir(parents=True)
        (path_train_imbalanced_2).mkdir(parents=True)
        for work in ['doctors', 'nurses']:
            (path_train_balanced / work).mkdir(parents=True)


    def copy_half_files(src, dest):
        all_files = os.listdir(src)
        half_files = random.sample(all_files, len(all_files) // 2)
        for file in half_files:
            shutil.copy2(src / file, dest / file)


    copy_half_files(path_train_full / 'doctors' / 'male', path_train_balanced / 'doctors' )
    copy_half_files(path_train_full / 'doctors' / 'female', path_train_balanced / 'doctors')
    copy_half_files(path_train_full / 'nurses' / 'male', path_train_balanced / 'nurses')
    copy_half_files(path_train_full / 'nurses' / 'female', path_train_balanced / 'nurses')


    shutil.copytree(path_train_full / 'doctors' / 'male', path_train_imbalanced_1 / 'doctors')
    shutil.copytree(path_train_full / 'doctors' / 'female', path_train_imbalanced_2 / 'doctors')
    shutil.copytree(path_train_full / 'nurses' / 'male', path_train_imbalanced_2 / 'nurses')
    shutil.copytree(path_train_full / 'nurses' / 'female', path_train_imbalanced_1 / 'nurses' )

def copy_test_dataset(path_dataset=Path.cwd() / 'data' / 'datasets' / 'doctor_nurse_2'):
    path_test_full = path_dataset / 'balanced_train_test' / 'test'

    path_train_test_split = path_dataset / 'train_test_split' 

    path_test_splits = [path_train_test_split / 'train_balanced' / 'test', path_train_test_split / 'train_imbalanced_1' / 'test', path_train_test_split / 'train_imbalanced_2' / 'test']
    for path_test in path_test_splits:
        if not path_test.exists():
            path_test.mkdir(parents=True)
        for work in ['doctors', 'nurses']:
            (path_test / work).mkdir(parents=True)
            for gender in ['male', 'female']:
                src = path_test_full / work / gender
                dest = path_test / work
                for file in os.listdir(src):
                    shutil.copy2(src / file, dest / file)
                    os.rename(dest / file, dest / f"{work}_{gender}_{file}")

def create_biased_dataset(bias_ratio: float, train_ratio: float, random_seed: int = 42):
    """
    Create a biased and a balanced dataset of doctor and nurse images with specified bias and train ratios.

    The function organizes the datasets into two main categories: 'doctors' and 'nurses', with further 
    subcategories of 'male' and 'female'. The biased dataset will have an overrepresentation of a specific 
    gender within each category based on the bias ratio, while the balanced dataset will have an equal 
    representation of both genders in each category. The training and testing sets are also created based on the train ratio.

    Parameters:
    bias_ratio (float): Ratio to determine the bias towards gender within each category in the biased dataset.
                        Should be a float between 0 and 1. 
                        If 0, the biased dataset will only contain male doctors and female nurses. 
                        If 1, the biased dataset will only contain female doctors and male nurses. 
                        If 0.5, the dataset will contain all of the train dataset, having thus same number of male/female for doctor/nurses.

    train_ratio (float): Ratio to split the original dataset into training and testing sets. 
                         Should be a float between 0 and 1, representing the proportion of the dataset to include in the train split.

    random_seed (int, optional): Seed for the random number generator for reproducibility. 
                                 Defaults to 42.

    Returns:
    None. The function creates the dataset folders and populates them with images from the original dataset.

    Note:
    The function assumes the existence of a balanced dataset of doctor and nurse images, categorized by gender.
    """
    def compute_number_samples_train_class(num_sample_class, ratio):
        def func(x):
            return x / (1-x)

        def func2(x):
            return func(1-x)
        
        if ratio <= 0.5:
            ratio_minority_class = func(ratio)
        else:
            ratio_minority_class = func2(ratio) 

        num_sample_minority_class = int(num_sample_class * ratio_minority_class)
        num_sample_majority_class = num_sample_class
        total_samples = num_sample_minority_class + num_sample_majority_class
        return num_sample_minority_class, int(total_samples)

    random.seed(random_seed)
    path_doctor_nurse = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse'
    path_biased_dataset_root = path_doctor_nurse / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'
    
    path_biased_train = path_biased_dataset_root / 'biased' / 'train'
    path_biased_test = path_biased_dataset_root / 'biased' /'test'
    
    path_balanced_train = path_biased_dataset_root / 'balanced' / 'train'
    path_balanced_test = path_biased_dataset_root / 'balanced' / 'test'

    path_test = path_biased_dataset_root / 'test'

    path_balanced_dataset = path_doctor_nurse / 'balanced_dataset'

    for path in [path_biased_train, path_biased_test, path_balanced_train, path_balanced_test]:
        for category in ['doctors', 'nurses']:
            os.makedirs(path / category, exist_ok=True)

    for category in ['doctors', 'nurses']:
        for gender in ['male', 'female']:
            os.makedirs(path_test / f'{category}_{gender}', exist_ok=True)

    # Count the number of doctors and nurses in the balanced dataset
    num_doctors_male, num_doctors_female, num_nurses_male, num_nurses_female = count_files_dataset(path_balanced_dataset, verbose=False)

    # Assert the balanced dataset is indeed balanced
    assert num_doctors_male == num_doctors_female == num_nurses_male == num_nurses_female
    
    # Compute the number of samples per class / gender for the testing dataset
    num_class_gender_test = math.floor(num_doctors_male * (1 - train_ratio))

    # Compute the number of samples per class per gender for the training dataset
    num_sample_minority_class, num_train_class = compute_number_samples_train_class(num_doctors_male, bias_ratio)
    num_train_class = math.floor(num_train_class * train_ratio)
    num_train_class_gender = int(num_train_class / 2)
    num_sample_minority_class = int(num_train_class - (num_doctors_female * train_ratio))
    print(f"Ratio {bias_ratio}, num_train_class {num_train_class}, num_train_class gender {num_train_class_gender}, num_sample_minority_class {num_sample_minority_class}")

     # Global list for test_files to avoid overlap with training sets
    test_files_global = []

    # Create test datasets
    for category in ['doctors', 'nurses']:
        for gender in ['male', 'female']:
            path_category_gender_balanced = path_balanced_dataset / category / gender
            files = list(path_category_gender_balanced.glob('*.*'))
            random.shuffle(files)
            test_files = files[:num_class_gender_test]
            test_files_global += test_files
            for file in test_files:
                new_filename = file.stem + '_' + gender + file.suffix
                shutil.copy2(file, path_biased_test / category / new_filename)
                shutil.copy2(file, path_balanced_test / category / new_filename)
                shutil.copy2(file, path_test / f'{category}_{gender}' / new_filename)

    # Create balanced training dataset
    for category in ['doctors', 'nurses']:
        for gender in ['male', 'female']:
            path_category_gender_balanced = path_balanced_dataset / category / gender
            files = [file for file in path_category_gender_balanced.glob('*.*') if file not in test_files_global]
            random.shuffle(files)
            train_files = files[:num_train_class_gender]
            for file in train_files:
                new_filename = file.stem + '_' + gender + file.suffix
                shutil.copy2(file, path_balanced_train / category / new_filename)
    
    # Create biased training dataset
    for category in ['doctors', 'nurses']:
        for gender in ['male', 'female']:
            path_category_gender_balanced = path_balanced_dataset / category / gender
            files = [file for file in path_category_gender_balanced.glob('*.*') if file not in test_files_global]
            random.shuffle(files)

            dominant_condition = (bias_ratio <= 0.5 and gender == 'male' and category == 'nurses') or \
                                 (bias_ratio <= 0.5 and gender == 'female' and category == 'doctors') or \
                                 (bias_ratio > 0.5 and gender == 'female' and category == 'nurses') or \
                                 (bias_ratio > 0.5 and gender == 'male' and category == 'doctors')
            
            if dominant_condition:
                train_files = files
            else:
                train_files = files[:num_sample_minority_class]
            for file in train_files:
                new_filename = file.stem + '_' + gender + file.suffix
                shutil.copy2(file, path_biased_train / category / new_filename)


if __name__ == '__main__':
    path_doctor_nurse = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse_2'
    #remove_ash_extensions_image()
    #transfer_original_to_gender_dataset()

    # count_files_dataset(path_doctor_nurse / 'original_dataset_gender') 
    #create_balanced_dataset(path_doctor_nurse)
    # count_files_dataset(path_doctor_nurse / 'balanced_dataset') 
    # create_balanced_train_test_set(ratio_train=0.75, path_root=path_doctor_nurse)
    create_train_dataset(path_doctor_nurse)
    copy_test_dataset()
    # count_files_train_test_set(path_doctor_nurse / 'default_train_test_75', True)
    # create_biased_dataset(bias_ratio=0, train_ratio=0.75, random_seed=42)
    # create_biased_dataset(bias_ratio=0.25, train_ratio=0.75, random_seed=42)
    # create_biased_dataset(bias_ratio=0.5, train_ratio=0.75, random_seed=42)
    # create_biased_dataset(bias_ratio=0.66, train_ratio=0.75, random_seed=42)
    # create_biased_dataset(bias_ratio=0.75, train_ratio=0.75, random_seed=42)
    # create_biased_dataset(bias_ratio=1, train_ratio=0.75, random_seed=42)
    
    # Note: I've copied the test set manually, sorry if anyone has to work on the project after, with my Thesis deadline it was the rush. 