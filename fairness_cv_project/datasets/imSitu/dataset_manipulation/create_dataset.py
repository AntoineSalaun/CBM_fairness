import os
import random
from pathlib import Path
import json
import shutil


def create_dataset(verbs, path_dataset):

    # Path to the directory containing the images
    image_dir = 'data/datasets/imSitu/original_data'

    # Path to the directory where the new folders will be created
    output_dir = path_dataset / 'original'

    # Loop through each verb
    for verb in verbs:
        # Create a new directory for the verb if it doesn't already exist
        verb_dir = os.path.join(output_dir, verb)
        if not os.path.exists(verb_dir):
            os.makedirs(verb_dir)
        
        # Loop through each image in the image directory
        for filename in os.listdir(image_dir):
            # Check if the filename starts with the verb
            if filename.startswith(verb):
                # Copy the image to the verb's directory
                src_path = os.path.join(image_dir, filename)
                dst_path = os.path.join(verb_dir, filename)
                shutil.copy(src_path, dst_path)

def create_targets(verbs, path_dataset, threshold_target_concepts_retained=10):
    """
    Creates the target_agent_count, target_concept_count and target_concepts_retained dictionaries
    """
    path_imsitu = Path('data/datasets/imSitu')
    full_targets = json.load(open(path_imsitu / 'metadata' / 'full.json'))

    selected_targets = {}
    for verb in verbs:
        selected_targets[verb] = full_targets[verb]

    target_concept_count = {}
    target_agent_count = {}
    for verb in selected_targets:
        
        target_concept_count[verb] = {}
        target_agent_count[verb] = {}


        for image in selected_targets[verb].values():
            image_agent = image['agent']
            image_concepts = image['concepts']
            
            if image_agent not in target_agent_count[verb]:
                target_agent_count[verb][image_agent] = 1
            else:
                target_agent_count[verb][image_agent] += 1

            for concept in image_concepts:
                if concept not in target_concept_count[verb]:
                    target_concept_count[verb][concept] = 1
                else:
                    target_concept_count[verb][concept] += 1
            
    target_concepts_retained = {}

    for verb in target_concept_count:
        target_concepts_retained[verb] = []
        for concept in target_concept_count[verb]:
            if target_concept_count[verb][concept] >= threshold_target_concepts_retained:
                target_concepts_retained[verb].append(concept)

    """
    print(target_agent_count)
    print('_'*50)
    print(target_concept_count)
    print('_'*50)
    print(target_concepts_retained)
    print(type(target_agent_count))
    print(type(target_concept_count))
    print(type(target_concepts_retained))
    """

    # Save target_agent_count dictionary to JSON
    with open(path_dataset / 'target_agent_count.json', 'w') as f:
        json.dump(target_agent_count, f)

    # Save target_concept_count dictionary to JSON
    with open(path_dataset / 'target_concept_count.json', 'w') as f:
        json.dump(target_concept_count, f)

    # Save target_concepts_retained dictionary to JSON
    with open(path_dataset / 'target_concepts_retained.json', 'w') as f:
        json.dump(target_concepts_retained, f)

    with open(path_dataset / 'target_original_metadata.json', 'w') as f:
        json.dump(selected_targets, f)

def select_images_with_gender(path_dataset):

    list_gender = ['man', 'male', 'woman', 'female']
    target_original_metadata = json.load(open(path_dataset / 'target_original_metadata.json'))

    entries_with_gender = {}

    for target, images in target_original_metadata.items():
        current_folder = path_dataset / 'original' / target
        path_dest = path_dataset / 'human_images' / 'datasets' /'original' / target
        entries_with_gender[target] = {}

        if not os.path.exists(path_dest):
            os.makedirs(path_dest)

        for image_name, metadata in images.items():
            if any(name in metadata['agent'] for name in list_gender):
                shutil.copy(current_folder / image_name, path_dest / image_name)
                entries_with_gender[target][image_name] = metadata
    
    with open(path_dataset / 'human_images' / 'metadata.json', 'w') as f:
        json.dump(entries_with_gender, f)

def select_images_with_gender_v2(path_dataset):
    """
    Put images with their target and respective gender
    """
    target_original_metadata = json.load(open(path_dataset / 'target_original_metadata.json'))
    list_male = ['man', 'male']
    list_female = ['woman', 'female']
    entries_with_gender = {}

    for target, images in target_original_metadata.items():
        current_folder = path_dataset / 'original' / target
        path_dest = path_dataset / 'human_images' / 'datasets' / 'original_with_gender' / target
        entries_with_gender[target] = {'male': {}, 'female': {}}

        if not os.path.exists(path_dest):
            os.makedirs(path_dest / 'male')
            os.makedirs(path_dest / 'female')

        for image_name, metadata in images.items():
            if any(name in metadata['agent'] for name in list_female):
                entries_with_gender[target]['female'][image_name] = metadata
                shutil.copy(current_folder / image_name, path_dest / 'female' / image_name)
            elif any(name in metadata['agent'] for name in list_male):
                entries_with_gender[target]['male'][image_name] = metadata
                shutil.copy(current_folder / image_name, path_dest / 'male' / image_name)

    # Metadata is same format as target original metadata: dict[target][image] = metadata
    # Metadata V2 is dict[target][gender][image] = metadata
    with open(path_dataset / 'human_images' / 'metadata_v2.json', 'w') as f:
        json.dump(entries_with_gender, f)

def train_test_split(path_dataset, test_size=0.25, random_seed=0):

    random.seed(random_seed)
    categories = os.listdir(path_dataset / 'human_images' / 'datasets' / 'original_with_gender_balanced')

    for category in categories:
        category_path = path_dataset / 'human_images' / 'datasets' / 'original_with_gender_balanced' / category

        male_images = os.listdir(category_path / 'male')
        female_images = os.listdir(category_path / 'female')

        male_train = random.sample(male_images, int(len(male_images) * (1 - test_size)))
        female_train = random.sample(female_images, int(len(female_images) * (1 - test_size)))

        male_test = list(set(male_images) - set(male_train))
        female_test = list(set(female_images) - set(female_train))

        train_path = path_dataset / 'human_images' / 'datasets' / 'train' / category
        test_path = path_dataset / 'human_images' / 'datasets' / 'test' / category
        test_with_gender_path = path_dataset / 'human_images' / 'datasets' / 'test_with_gender' / category 

        if not os.path.exists(train_path):
            os.makedirs(train_path / 'male')
            os.makedirs(train_path / 'female')

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        if not os.path.exists(test_with_gender_path):
            os.makedirs(test_with_gender_path / 'male')
            os.makedirs(test_with_gender_path / 'female')
        
        for image in male_train:
            base, ext = os.path.splitext(image)
            filename = f'{base}_male{ext}'
            shutil.copy(category_path / 'male' / image, train_path / 'male' / filename)
        for image in female_train:
            base, ext = os.path.splitext(image)
            filename = f'{base}_female{ext}' 
            shutil.copy(category_path / 'female' / image, train_path / 'female' / filename)
        for image in male_test:
            base, ext = os.path.splitext(image)
            filename = f'{base}_male{ext}'
            shutil.copy(category_path / 'male' / image, test_path / filename)
            shutil.copy(category_path / 'male' / image, test_with_gender_path / 'male' / filename)
        for image in female_test:
            base, ext = os.path.splitext(image)
            filename = f'{base}_female{ext}' 
            shutil.copy(category_path / 'female' / image, test_path / filename) 
            shutil.copy(category_path / 'female' / image, test_with_gender_path / 'female' / filename) 


def create_copy_with_suffix(suffix):
    def copy_with_suffix(src, dst):
        base, ext = os.path.splitext(dst)
        dst = f"{base}_{suffix}{ext}"
        shutil.copy2(src, dst)
    return copy_with_suffix


# Works only for binary dataset
def create_train_dataset(path_dataset, random_seed=0):

    random.seed(random_seed)
    categories = os.listdir(path_dataset / 'human_images' / 'datasets' / 'original_with_gender_balanced')

    path_train_balanced = path_dataset / 'human_images' / 'datasets' / 'train_balanced'
    path_train_imbalanced_1 = path_dataset / 'human_images' / 'datasets' / 'train_imbalanced_1'
    path_train_imbalanced_2 =  path_dataset / 'human_images' / 'datasets' / 'train_imbalanced_2'

    for i, category in enumerate(categories):
        path_category = path_dataset / 'human_images' / 'datasets' / 'train' / category

        for gender in ['male', 'female']:
            path_category_gender = path_category / gender

            if not os.path.exists(path_train_balanced / category):
                os.makedirs(path_train_balanced / category)
        
            
            files = os.listdir(path_category_gender)
            num_files = len(files)
            train_balanced = random.sample(files, int(num_files // 2)) 

            for file in train_balanced:
                shutil.copy(path_category_gender / file, path_train_balanced / category / file)
            
            if i == 0:
                if gender == 'male':
                    shutil.copytree(path_category_gender, path_train_imbalanced_1 / category)
                    with open(path_dataset / 'human_images' / 'class_male_1.txt', 'w') as f:
                        f.write(f'{category}\n')
                else:
                    shutil.copytree(path_category_gender, path_train_imbalanced_2 / category)
            else:
                if gender == 'female':
                    shutil.copytree(path_category_gender, path_train_imbalanced_1 / category)
                else:
                    shutil.copytree(path_category_gender, path_train_imbalanced_2 / category)



def create_balanced_dataset(path_dataset):

    target_metadata = json.load(open(path_dataset / 'human_images' / 'metadata_v2.json'))

    lowest_number = get_lowest_number(path_dataset)

    path_dataset_balanced = path_dataset / 'human_images' / 'datasets' /'original_with_gender_balanced'

    # Put images in the destination folder 
    # Take a balanced version, i.e. with the same number for every target and gender
    # First destination has also gender, second destination has only targets
    for target in target_metadata.keys():
        for gender in ['male', 'female']:
            
            # The sort take '101' before '99' but it's not a problem
            path_src = path_dataset / 'human_images' /'datasets' / 'original_with_gender' / target /  gender
            image_list = os.listdir(path_src)
            image_list.sort()
            
            path_dest = path_dataset_balanced / target / gender
            
            if not os.path.exists(path_dest):
                os.makedirs(path_dest)
            
            for i in range(lowest_number):
                shutil.copy(path_src / image_list[i] , path_dest / image_list[i])


def get_lowest_number(path_dataset):
    target_metadata = json.load(open(path_dataset / 'human_images' /'metadata_v2.json'))

    lowest_number = 99999
    for target, entries in target_metadata.items():
        for gender, images in entries.items():
            if len(images) < lowest_number:
                lowest_number = len(images)
    
    return lowest_number


def transfer_to_train_test_split(path_dataset):
    path_test = path_dataset / 'human_images' / 'datasets' / 'test'
    path_train_balanced = path_dataset / 'human_images' / 'datasets' / 'train_balanced'
    path_train_imbalanced_1 = path_dataset / 'human_images' / 'datasets' / 'train_imbalanced_1'
    path_train_imbalanced_2 = path_dataset / 'human_images' / 'datasets' / 'train_imbalanced_2'

    path_test_with_gender_original = path_dataset / 'human_images' / 'datasets' / 'test_with_gender'
    path_test_with_gender_new = path_dataset / 'human_images' / 'test'

    path_train_test_split = path_dataset / 'human_images' / 'train_test_split'

    for category in os.listdir(path_test):
        shutil.copytree(path_test / category, path_train_test_split / 'balanced' / 'test' / category)
        shutil.copytree(path_test / category, path_train_test_split / 'imbalanced_1' / 'test' / category)
        shutil.copytree(path_test / category, path_train_test_split / 'imbalanced_2' / 'test' / category)

        shutil.copytree(path_train_balanced / category, path_train_test_split / 'balanced' / 'train' / category)
        shutil.copytree(path_train_imbalanced_1 / category, path_train_test_split / 'imbalanced_1' / 'train' / category)
        shutil.copytree(path_train_imbalanced_2 / category, path_train_test_split / 'imbalanced_2' / 'train' / category)
        for gender in ['male', 'female']:
           shutil.copytree(path_test_with_gender_original / category / gender, path_test_with_gender_new / f'{category}_{gender}') 


def pipeline(verbs, dataset_name):
    path_dataset = Path('data/datasets/imSitu/data') / dataset_name
    create_dataset(verbs, path_dataset)
    create_targets(verbs, path_dataset, 10)
    select_images_with_gender(path_dataset)
    select_images_with_gender_v2(path_dataset)
    create_balanced_dataset(path_dataset)
    train_test_split(path_dataset)
    create_train_dataset(path_dataset)
    transfer_to_train_test_split(path_dataset)
    # create_imbalanced_datasets_binary(path_dataset)
    


phoning_eating = ['phoning', 'eating']
dataset_name = 'phoning_eating'
pipeline(phoning_eating, dataset_name)

#path_dataset = Path('data/datasets/imSitu/data/phoning_cooking')

"""
verbs = ['cooking', 'driving', 'cleaning', 'phoning']
phoning_eating = ['phoning', 'eating']
create_dataset(phoning_eating)
create_targets(phoning_eating, 10)
"""
# select_images_with_gender_v2(path_dataset)
# create_balanced_dataset(Path('data/datasets/imSitu/data/phoning_cooking/human_images'))
# create_imbalanced_datasets_binary(Path('data/datasets/imSitu/data/phoning_cooking/human_images'))