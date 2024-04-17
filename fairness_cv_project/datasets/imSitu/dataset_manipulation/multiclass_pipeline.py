import os
import random
from pathlib import Path
import json
import shutil

def create_original_data_and_metadata(verbs, path_imsitu, path_dataset):
    """
    Create a folder with the original data and a metadata file for each verb in the verb list 
    It filters the images that don't have an human in it
    """
    original_metadata = json.load(open(path_imsitu / 'metadata' / 'full.json'))
    path_original_data = path_imsitu / 'original_data'

    selected_targets = {}

    for verb in verbs:
        selected_targets[verb] = original_metadata[verb]

    target_concepts_count = {}
    metadata = {}
    metadata_by_gender = {}

    list_male = ['male', 'man']
    list_female = ['female', 'woman']
    list_gender = ['male', 'man', 'female', 'woman']

    # Loop through verbs
    for verb in selected_targets:

        target_concepts_count[verb] = {}
        metadata[verb] = {}
        metadata_by_gender[verb] = {'male': {}, 'female': {}}

        path_dest = path_dataset / 'original' / verb
        os.makedirs(path_dest / 'male') 
        os.makedirs(path_dest / 'female') 

        for image_id, original_metadata in selected_targets[verb].items():

            if any(name in original_metadata['agent'] for name in list_gender):

                image_agent = original_metadata['agent']

            image_concepts = original_metadata['concepts']
            metadata[verb][image_id] = original_metadata 

            for concept in image_concepts:
                if concept not in target_concepts_count[verb]:
                    target_concepts_count[verb][concept] = 1
                else:
                    target_concepts_count[verb][concept] += 1

            # Depending on agent gender, copy metadata and file
            if any(name in original_metadata['agent'] for name in list_female):
                metadata_by_gender[verb]['female'][image_id] = original_metadata
                shutil.copy(path_original_data / image_id, path_dest / 'female')

            elif any(name in original_metadata['agent'] for name in list_male):
                metadata_by_gender[verb]['male'][image_id] = original_metadata
                shutil.copy(path_original_data / image_id, path_dest / 'male')


    with open(path_dataset / 'metadata.json', 'w') as f:
        json.dump(metadata, f)

    with open(path_dataset / 'metadata_by_gender.json', 'w') as f:
        json.dump(metadata_by_gender, f)

    with open(path_dataset / 'target_concepts_count.json', 'w') as f: 
        json.dump(target_concepts_count, f)
        
def filter_concepts(path_dataset, threshold_target_concepts_retained):
    """
    Take the concepts that are the most present for each class
    """
    target_concepts_count = json.load(open(path_dataset / 'target_concepts_count.json'))

    target_concepts_retained = {}

    for verb in target_concepts_count:
        target_concepts_retained[verb] = []
        for concept in target_concepts_count[verb]:
            if target_concepts_count[verb][concept] >= threshold_target_concepts_retained:
                target_concepts_retained[verb].append(concept)

    with open(path_dataset / 'target_concepts_retained.json', 'w') as f:
        json.dump(target_concepts_retained, f)

def balance_dataset(path_dataset):
    """
    Create a folder with the balanced data and a metadata file for each verb in the verb list
    It is balanced w.r.t gender, i.e. for each class, you take the lowest # of samples between
    males and females 
    """
    metadata_with_gender = json.load(open(path_dataset / 'metadata_by_gender.json'))

    path_dataset_original = path_dataset / 'original'
    path_dataset_balanced = path_dataset / 'full_balanced'

    metadata_full_balanced = {}

    for target, value in metadata_with_gender.items():
        min_samples = min(len(value['male']), len(value['female']))
        metadata_full_balanced[target] = {'male': [], 'female': []}

        # Note that sort will take '101' before '99'
        for gender in ['male', 'female']:
            path_src = path_dataset_original / target / gender
            image_list = os.listdir(path_src)
            image_list.sort()

            path_dest = path_dataset_balanced / target / gender

            if not os.path.exists(path_dest):
                os.makedirs(path_dest)

            for i in range(min_samples):
                shutil.copy(path_src / image_list[i], path_dest / image_list[i])
                metadata_full_balanced[target][gender].append(metadata_with_gender[target][gender][image_list[i]])

    with open(path_dataset / 'metadata_full_balanced.json', 'w') as f: 
        json.dump(metadata_full_balanced, f)


def take_half_verbs(path_dataset):
    """
    Take half of the concepts and put them in a file
    This half is the first half of the sorted list of concepts
    """
    metadata = json.load(open(path_dataset / 'metadata.json'))

    list_target = sorted(metadata.keys())

    cutoff = int(len(list_target) / 2)

    first_half = list_target[:cutoff]
    second_half = list_target[cutoff:]
    
    with open(path_dataset / 'verb_group_1.txt', 'w') as f:
        for verb in first_half:
            f.write(str(verb) + "\n")
        
    with open(path_dataset / 'verb_group_2.txt', 'w') as f:
        for verb in second_half:
            f.write(str(verb) + "\n")


def train_test_split_full_dataset(path_dataset, test_size=0.25, random_seed=0):
    """
    Split the full dataset into train and test
    This full dataset will be engineered at a later stage
    """
    random.seed(random_seed)
    
    path_balanced = path_dataset / 'full_balanced'
    targets = os.listdir(path_balanced)
    targets.sort() 
    for target in targets:
        path_target = path_balanced / target
        
        male_images = os.listdir(path_target / 'male')
        female_images = os.listdir(path_target / 'female')
        
        male_train = random.sample(male_images, int(len(male_images) * (1 - test_size)))
        female_train = random.sample(female_images, int(len(female_images) * (1 - test_size)))

        male_test = list(set(male_images) - set(male_train))
        female_test = list(set(female_images) - set(female_train))
        
        path_train = path_dataset / 'train' / 'train_full' / target
        path_test = path_dataset / 'test' / target
        path_test_with_gender = path_dataset / 'test_with_gender' / target

        if not os.path.exists(path_train):
            os.makedirs(path_train / 'male')
            os.makedirs(path_train / 'female')
            
        if not os.path.exists(path_test_with_gender):
            os.makedirs(path_test_with_gender / 'male')
            os.makedirs(path_test_with_gender / 'female')
            
        if not os.path.exists(path_test):
            os.makedirs(path_test)
            
        for image in male_train:
            base, ext = os.path.splitext(image)
            filename = f'{base}_male{ext}'
            shutil.copy(path_target / 'male' / image, path_train / 'male' / filename)
            
        for image in female_train:
            base, ext = os.path.splitext(image)
            filename = f'{base}_female{ext}' 
            shutil.copy(path_target / 'female' / image, path_train / 'female' / filename)
            
        for image in male_test:
            base, ext = os.path.splitext(image)
            filename = f'{base}_male{ext}'
            shutil.copy(path_target / 'male' / image, path_test / filename)
            shutil.copy(path_target / 'male' / image, path_test_with_gender / 'male' / filename)
            
        for image in female_test:
            base, ext = os.path.splitext(image)
            filename = f'{base}_female{ext}' 
            shutil.copy(path_target / 'female' / image, path_test / filename) 
            shutil.copy(path_target / 'female' / image, path_test_with_gender / 'female' / filename) 
        

def train_test_val_split_full_dataset(path_dataset, test_size=0.2, val_size=0.2, random_seed=0, balanced=True):
    """
    Split the full dataset into train, test, and validation sets.
    This full dataset will be engineered at a later stage.
    """
    random.seed(random_seed)
    
    path_balanced = Path(path_dataset) / 'full_balanced' if balanced else Path(path_dataset) / 'original'
    targets = os.listdir(path_balanced)
    targets.sort()
    for target in targets:
        path_target = path_balanced / target
        
        male_images = os.listdir(path_target / 'male')
        female_images = os.listdir(path_target / 'female')
        
        male_train = random.sample(male_images, int(len(male_images) * (1 - test_size - val_size)))
        female_train = random.sample(female_images, int(len(female_images) * (1 - test_size - val_size)))
        
        male_remain = list(set(male_images) - set(male_train))
        female_remain = list(set(female_images) - set(female_train))

        male_val = random.sample(male_remain, int(len(male_remain) * val_size / (val_size + test_size)))
        female_val = random.sample(female_remain, int(len(female_remain) * val_size / (val_size + test_size)))
        
        male_test = list(set(male_remain) - set(male_val))
        female_test = list(set(female_remain) - set(female_val))

        path_train = Path(path_dataset) / 'train' / 'train_full' / target
        path_val = Path(path_dataset) / 'val' / target
        path_test = Path(path_dataset) / 'test' / target
        path_test_with_gender = Path(path_dataset) / 'test_with_gender' / target

        if not os.path.exists(path_train):
            os.makedirs(path_train / 'male')
            os.makedirs(path_train / 'female')

        if not os.path.exists(path_val):
            os.makedirs(path_val)

        if not os.path.exists(path_test_with_gender):
            os.makedirs(path_test_with_gender / 'male')
            os.makedirs(path_test_with_gender / 'female')

        if not os.path.exists(path_test):
            os.makedirs(path_test)

        copy_files(male_train, 'male', path_target, path_train, True)
        copy_files(female_train, 'female', path_target, path_train, True)

        copy_files(male_val, 'male', path_target, path_val, False)
        copy_files(female_val, 'female', path_target, path_val, False)

        copy_files(male_test, 'male', path_target, path_test, False)
        copy_files(female_test, 'female', path_target, path_test, False)

        copy_files(male_test, 'male', path_target, path_test_with_gender, True)
        copy_files(female_test, 'female', path_target, path_test_with_gender, True)


def train_val_full(path_dataset):
    path_train = path_dataset / 'train' / 'train_full'
    path_val = path_dataset / 'val'
    targets = os.listdir(path_val)
    targets.sort()

    path_train_val = path_dataset / 'train_val'
    
    for target in targets:
        path_train_target = path_train_val / 'train' / target
        path_val_target = path_train_val / 'val' / target

        if not os.path.exists(path_train_target):
            os.makedirs(path_train_target)

        if not os.path.exists(path_val_target):
            os.makedirs(path_val_target)
        
        files_val = os.listdir(path_val / target)
        for file in files_val:
            shutil.copy(path_val / target / file, path_val_target / file)
        # Copy files from path_val / target to path_val_target
        for gender in ['male', 'female']:
            
            files_train = os.listdir(path_train / target / gender)
            for file in files_train:
                shutil.copy(path_train / target / gender / file, path_train_target / file)


def copy_files(image_list, gender, source_folder, destination_folder, gender_folder=False):
    for image in image_list:
        base, ext = os.path.splitext(image)
        filename = f'{base}_{gender}{ext}'

        if gender_folder:
            shutil.copy(source_folder / gender / image, destination_folder / gender / filename)
        else:
            shutil.copy(source_folder / gender / image, destination_folder / filename)


def sample_half_train(path_dataset, random_seed = 0):
    """
    Sample half of the train set and put it in a new folder
    We sample male and female images to have a balanced train set 
    """
    random.seed(random_seed)
    
    targets = os.listdir(path_dataset / 'test')
    path_train_full = path_dataset / 'train' / 'train_full'
    path_train_half = path_dataset / 'train' / 'train_half'
    
    for target in targets:
        os.makedirs(path_train_half / target)
        
        for gender in ['male', 'female']:
            files = os.listdir(path_train_full / target / gender)
            num_files = len(files)
            train_half = random.sample(files, num_files // 2)
            
            for file in train_half:
                shutil.copy(path_train_full / target / gender / file, path_train_half / target / file)
                

def add_val_to_balanced(path_dataset):
    path_val = path_dataset / 'val'
    path_train_val_split = path_dataset / 'train_val_split'

    path_train_balanced = path_train_val_split / 'train_balanced' 

    for target in os.listdir(path_val):
        shutil.copytree(path_val/ target, path_train_balanced / 'val'/ target)
    
def train_val_split_functioning_dataset(path_dataset):
    """
    Split the dataset into three datasets that could be used to train and test a model

    Every dataset has the same test set, but a different train set:

    1) Balanced train, with half man half woman
    2) Imbalanced train 1: verbs in the list have only male samples, others have only female
    3) Imbalanced train 2: opposite of imbalanced train 1

    Note that the 3 train sets have the same number of samples per class
    """
    path_train_full = path_dataset / 'train' / 'train_full'
    
    path_val = path_dataset / 'val'
    
    path_train_half = path_dataset / 'train' / 'train_half'
     
    path_train_val_split = path_dataset / 'train_val_split'

    path_train_balanced = path_train_val_split / 'train_balanced'
    path_train_imbalanced_1 = path_train_val_split / 'train_imbalanced_1'
    path_train_imbalanced_2 = path_train_val_split / 'train_imbalanced_2'
    
    with open(path_dataset / 'verb_group_1.txt', 'r') as f:
        verbs_group_1 = [line.strip() for line in f.readlines()]
        
    for target in os.listdir(path_val):
        shutil.copytree(path_val/ target, path_train_imbalanced_1 / 'val'/ target)
        shutil.copytree(path_val/ target, path_train_imbalanced_2 / 'val'/ target)
        shutil.copytree(path_val/ target, path_train_balanced / 'val'/ target)
        
        shutil.copytree(path_train_half / target, path_train_balanced / 'train' / target)
        
        if target in verbs_group_1:
            shutil.copytree(path_train_full / target / 'male', path_train_imbalanced_1 / 'train' / target)
            shutil.copytree(path_train_full / target / 'female', path_train_imbalanced_2 / 'train' / target)
        else:
            shutil.copytree(path_train_full / target / 'female', path_train_imbalanced_1 / 'train' / target)
            shutil.copytree(path_train_full / target / 'male', path_train_imbalanced_2 / 'train' / target)
            
def pipeline(list_verbs, path_imsitu, path_dataset, threshold_target_concepts_retained=15):
    #create_original_data_and_metadata(list_verbs, path_imsitu, path_dataset)
    #filter_concepts(path_dataset, threshold_target_concepts_retained)
    #balance_dataset(path_dataset)
    #take_half_verbs(path_dataset)
    train_test_val_split_full_dataset(path_dataset)
    sample_half_train(path_dataset)
    train_val_split_functioning_dataset(path_dataset)

def pipeline_unbalanced(list_verbs, path_imsitu, path_dataset, threshold_target_concepts_retained=15):
    create_original_data_and_metadata(list_verbs, path_imsitu, path_dataset)
    filter_concepts(path_dataset, threshold_target_concepts_retained)
    take_half_verbs(path_dataset)
    train_test_val_split_full_dataset(path_dataset, balanced=False)
    train_val_full(path_dataset)

string_verbs = """
phoning
hugging
eating
admiring
leaning
putting
carrying
reading
communicating
hitchhiking
resting
vacuuming
talking
cleaning
sitting
patting
hunching
smelling
rehabilitating
perspiring
shushing
helping
practicing
pinning
embracing
drinking
pushing
crying
rubbing
feeding
shouting
tilting
spying
grinning
picking
caressing
licking
interviewing
driving
shivering
milking
grimacing
shelving
checking
coughing
manicuring
lifting
crouching
kissing
gasping
instructing
ignoring
covering
sleeping
smiling
measuring
riding
tasting
stooping
calling
biting
stroking
jogging
distributing
washing
giving
asking
stuffing
tickling
stripping
heaving
swinging
winking
browsing
encouraging
arranging
hanging
squeezing
mopping
photographing
complaining
baking
brushing
walking
wiping
standing
chewing
scrubbing
scratching
wheeling
kneeling
stretching
snuggling
shrugging
telephoning
staring
working
pinching
buying
dialing
kicking
sniffing
opening
speaking
cooking
waving
studying
slapping
slouching
frowning
bothering
praying
adjusting
buttoning
sweeping
applying
yanking
climbing
signaling
displaying
pouting
sneezing
twirling
recovering
stirring
scooping
making
whistling
paying
recuperating
typing
operating
providing
weeping
shopping
glaring
emptying
wrinkling
exercising
strapping
crowning
giggling
running
falling
squinting
gardening
raking
weighing
smashing
tripping
serving
yawning
autographing
rocking
releasing
immersing
writing
smearing
shooting
vaulting
laughing
crafting
packaging
counting
placing
tying
cramming
dragging
saying
pumping
hoeing
twisting
buckling
pouring
painting
lathering
packing
wringing
combing
taping
tearing
pasting
frying
confronting
unpacking
turning
splashing
saluting
waiting
offering
misbehaving
potting
pedaling
pressing
jumping
reassuring
diving
skating
scolding
begging
"""



if __name__ == "__main__":
    dataset_name = '200_verbs'
    path_imSitu = Path('data/datasets/imSitu')
    path_dataset = Path('data/datasets/imSitu/data') / dataset_name
    threshold_target_concepts_retained = 15

    list_verbs = string_verbs.strip().split("\n")
    list_verbs_200 = string_verbs.strip().split("\n")
    list_verbs_200.sort()
    add_val_to_balanced(path_dataset)
    # pipeline(list_verbs_200, path_imSitu, path_dataset, threshold_target_concepts_retained)
    # pipeline_unbalanced(list_verbs_200, path_imSitu, path_dataset, threshold_target_concepts_retained)