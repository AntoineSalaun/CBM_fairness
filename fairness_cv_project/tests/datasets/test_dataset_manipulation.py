import pytest
from itertools import product
from pathlib import Path
import os

bias_ratios = [0, 0.25, 0.33, 0.5, 0.66, 0.75, 1]
train_ratios = [0.75]

# Create a list of tuples from bias_ratios and train_ratios
params = list(product(bias_ratios, train_ratios))

@pytest.mark.parametrize("bias_ratio, train_ratio", params)
def test_same_test_files_between_biased_and_balanced(bias_ratio: float, train_ratio: float):
    """
    This test checks that the same files are present in the test folders of the biased and balanced datasets.
    """
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse' / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'
    
    path_biased_test = path_root / 'biased' / 'test'
    path_balanced_test = path_root / 'balanced' / 'test'

    for category in ['doctors', 'nurses']:
        biased_files = set(os.listdir(path_biased_test / category))
        balanced_files = set(os.listdir(path_balanced_test / category))
        assert biased_files == balanced_files

@pytest.mark.parametrize("bias_ratio, train_ratio", params)
def test_same_file_between_test_biased_and_balanced(bias_ratio: float, train_ratio: float):
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse' / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'

    path_biased_test = path_root / 'biased' / 'test'
    path_balanced_test = path_root / 'balanced' / 'test'
    path_test = path_root / 'test'

    for category in ['doctors', 'nurses']:
        for gender in ['male', 'female']:
            biased_files = os.listdir(path_biased_test / category)
            set_biased_files = set([file for file in biased_files if file.split('.')[-2].endswith(f'_{gender}')])

            balanced_files = os.listdir(path_balanced_test / category)
            set_balanced_files = set([file for file in balanced_files if file.split('.')[-2].endswith(f'_{gender}')])

            set_test_files = set(os.listdir(path_test / f'{category}_{gender}'))
            assert set_biased_files == set_balanced_files == set_test_files

@pytest.mark.parametrize("bias_ratio, train_ratio", params)
def test_balanced_test_files(bias_ratio: float, train_ratio: float):
    """
    Check that test files have balanced gender ratios.
    """
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse' / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'
    
    path_test = path_root / 'balanced' / 'test'  # You can pick either 'biased' or 'balanced'

    for category in ['doctors', 'nurses']:
        files = os.listdir(path_test / category)
        num_male = len([file for file in files if file.split('.')[-2].endswith('_male')])
        num_female = len([file for file in files if file.split('.')[-2].endswith('_female')])
        assert num_male == num_female


@pytest.mark.parametrize("bias_ratio, train_ratio", params)
def test_same_number_train_samples_between_biased_and_balanced(bias_ratio: float, train_ratio: float):
    """
    This test checks that the number of training samples is the same between the biased and balanced datasets.
    """
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse' / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'
    
    path_biased_train = path_root / 'biased' / 'train'
    path_balanced_train = path_root / 'balanced' / 'train'

    for category in ['doctors', 'nurses']:
        num_biased_files = len(os.listdir(path_biased_train / category))
        num_balanced_files = len(os.listdir(path_balanced_train / category))
        # With the math.floor, we can have a difference of one sample in the dataset
        # assert num_balanced_files == num_biased_files
        assert abs(num_balanced_files - num_biased_files) <= 1


@pytest.mark.parametrize("bias_ratio, train_ratio", params)
def test_balanced_train_files(bias_ratio: float, train_ratio: float):
    """
    Check balanced train files are balanced 
    """
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse' / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'
    
    path_balanced_train = path_root / 'balanced' / 'train'

    for category in ['doctors', 'nurses']:
        files = os.listdir(path_balanced_train / category)
        num_male = len([file for file in files if file.split('.')[-2].endswith('_male')])
        num_female = len([file for file in files if file.split('.')[-2].endswith('_female')])
        # With the math.floor, we can have a difference of one sample in the dataset
        assert num_male == num_female


@pytest.mark.parametrize("bias_ratio, train_ratio", params)
def test_biased_train_files_follow_ratio(bias_ratio: float, train_ratio: float):
    """
    Check the biased train files follow the ratio
    """
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse' / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'
    
    path_biased_train = path_root / 'biased' / 'train'

    for category in ['doctors', 'nurses']:
        files = os.listdir(path_biased_train / category)
        num_male = len([file for file in files if file.split('.')[-2].endswith('_male')])
        num_female = len([file for file in files if file.split('.')[-2].endswith('_female')])
        total_samples = num_male + num_female
        if bias_ratio <= 0.5:
            dominant_gender = 'female' if category == 'doctors' else 'male'
            bias_ratio_complement = bias_ratio
        else:
            dominant_gender = 'male' if category == 'doctors' else 'female'
            bias_ratio_complement = 1 - bias_ratio
        
        dominant_samples = num_female if dominant_gender == 'female' else num_male
        non_dominant_samples = total_samples - dominant_samples
        print(f'Category {category}, bias_ratio {bias_ratio}, Dominant gender {dominant_gender} has dominant samples {dominant_samples} and non-dominant samples {non_dominant_samples}')

        # Here, we're asserting that the ratio of non-dominant samples to total samples is approximately equal to the bias_ratio.
        # You may need to adjust the precision of this check (1e-2) depending on your use case.
        assert abs(non_dominant_samples / total_samples - bias_ratio_complement) <= 1e-2

@pytest.mark.parametrize("bias_ratio, train_ratio", params)
def test_no_overlap_between_training_and_testing(bias_ratio: float, train_ratio: float):
    path_root = Path.cwd() / 'data' / 'datasets' / 'doctor_nurse' / f'biased_dataset_bias_{int(bias_ratio*100)}_train_{int(train_ratio*100)}'
    
    path_biased_train = path_root / 'biased' / 'train'
    path_balanced_train = path_root / 'balanced' / 'train'
    path_test = path_root / 'biased' / 'test'  # You can pick either 'biased' or 'balanced'

    for category in ['doctors', 'nurses']:
        train_files = set(os.listdir(path_balanced_train / category) + os.listdir(path_biased_train / category))
        test_files = set(os.listdir(path_test / category))
        assert len(train_files & test_files) == 0


"""
def super_test(bias_ratio: float, train_ratio: float):
    test_same_test_files_between_biased_and_balanced(bias_ratio, train_ratio)
    test_balanced_test_files(bias_ratio, train_ratio)
    test_same_number_train_samples_between_biased_and_balanced(bias_ratio, train_ratio)
    test_balanced_train_files(bias_ratio, train_ratio)
    test_biased_train_files_follow_ratio(bias_ratio, train_ratio)
    test_no_overlap_between_training_and_testing(bias_ratio, train_ratio)
"""