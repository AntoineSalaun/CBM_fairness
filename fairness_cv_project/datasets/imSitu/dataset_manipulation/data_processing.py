import json
import statistics
import random 
import copy

import pandas as pd
# Random seed for reproducibility, when selecting only one concept per noun
random.seed(1)

def read_imsitu_space():
    """
    Reads the imsitu_space.json file, which contains the mapping between the nouns and the concepts
    """
    imsitu = json.load(open("imsitu_space.json"))

    nouns = imsitu["nouns"]
    verbs = imsitu["verbs"]
    """
    random_key = random.choice(list(nouns.keys()))
    random_value = nouns[random_key]

    print(random_key)
    print(random_value)
    """

    return nouns, verbs


def take_nouns_mapping(nouns):
    """
    Takes the mapping between the nouns and the concepts, and returns a dictionary with the nouns as keys and the concepts as values
    """
    mapping = {}
    for key, value in nouns.items():
        # Values have synonyms - we take only one concept per noun, at random
        random_value = random.choice(value['gloss'])
        mapping[key] = random_value

    return mapping

def read_imsitu(file_name="train.json"):
    """
    Reads the imsitu dataset, and returns a dictionary with the image names as keys and the data as values
    """
    dataset = json.load(open(file_name))
    
    """
    random_key = random.choice(list(dataset.keys()))
    random_value = dataset[random_key]

    print(random_key)
    print(random_value)
    """    
    return dataset


def process_datapoint(entry, mapping):
    """
    Processes a datapoint from the imsitu dataset, and returns a dictionary with the target verb, the agent and the concepts
    """
    # Initialize variables for the target verb and agent
    agents = []

    target_verb = entry['verb']

    # Iterate over the frames in the data
    concepts = []

    for frame in entry['frames']:

        if 'agent' in frame:
            current_agent = frame['agent']
        elif 'agenttype' in frame:
            current_agent = frame['agenttype']
        else:
            current_agent = 'NA'

        agents.append(current_agent)

        current_concepts = list(frame.values())

        if current_agent != "NA":
            current_concepts.remove(current_agent)
        
        concepts.extend(current_concepts)

    #print(agents)
    # Find the most common agent 
    agent = statistics.mode(agents) 
    if agent == '':
        agent = 'NA'
    # Remove duplicates
    concepts = set(concepts)
    concepts.discard('')

    # Map the concepts to the nouns
    agent = mapping[agent] if agent != 'NA' else 'NA'
    concepts = [mapping[concept] for concept in concepts] 

    # Create the modified dictionary
    modified_dict = {'target': target_verb, 'agent': agent, 'concepts': concepts}

    return modified_dict

def process_dataset(dataset, mapping):
    classes_dict = {}
    
    for image_name, image_data in dataset.items():
        # Retrieving the class name, from images that are like "pressing_0001.jpg"
        class_name = image_name.split("_")[0] 
        
        # Creating the class in the dictionary
        if class_name not in classes_dict.keys():
            classes_dict[class_name] = {}
        
        classes_dict[class_name][image_name] = process_datapoint(image_data, mapping)
    


    return classes_dict

def merge_dataset(train, dev, test):
    """
    Merges the train, dev and test datasets into one
    """
    merged = copy.deepcopy(train)
    for class_name in train.keys():
        merged[class_name].update(dev[class_name])
        merged[class_name].update(test[class_name])
    return merged

def count_gender_per_class(path_full):
    full = json.load(open(path_full))

    df = pd.DataFrame(columns=['class_name', 'male', 'female', 'minimum', 'original_size'])
    male_name = ['man', 'male']
    female_name = ['woman', 'female']

    for class_name in full.keys():
        count_male = 0
        count_female = 0
        original_size = len(full[class_name])
        for image_name, image_metadata in full[class_name].items():
            if image_metadata['agent']:
                if any(name in image_metadata['agent'] for name in female_name):
                    count_female += 1
                elif any(name in image_metadata['agent'] for name in male_name):
                    count_male += 1
        df.loc[len(df)] = [class_name, count_male, count_female, min(count_female, count_male), original_size]
    
    df = df.sort_values(by='minimum', ascending=False) 
    print(df)
    # df.to_csv('data/datasets/imSitu/metadata/count_gender.csv')

count_gender_per_class('data/datasets/imSitu/data/phoning_cooking/only_gender_image/metadata.json')

"""
# pushing_176.jpg
example = {'frames': [{'item': 'n03931044', 'place': '', 'agent': 'n10287213'}, {'item': 'n03720163', 'place': 'n08588294', 'agent': 'n10287213'}, {'item': 'n03210940', 'place': '', 'agent': 'n10287213'}], 'verb': 'pressing'}

# slithering_83.jpg
example_2 = {'frames': [{'place': 'n08569998', 'agent': 'n01726692'}, {'place': 'n12102133', 'agent': 'n01726692'}, {'place': 'n09334396', 'agent': 'n01726692'}], 'verb': 'slithering'}

nouns, verbs = read_imsitu_space()
mapping = take_nouns_mapping(nouns)

train = read_imsitu()
train_classes = process_dataset(train, mapping)

with open('train_clean.json', 'w') as f:
    json.dump(train_classes, f)

test = read_imsitu("test.json")
test_classes = process_dataset(test, mapping)

with open('test_clean.json', 'w') as f:
    json.dump(test_classes, f)

dev = read_imsitu("dev.json")
dev_classes = process_dataset(dev, mapping)

with open('dev_clean.json', 'w') as f:
    json.dump(dev_classes, f)

merged = merge_dataset(train_classes, dev_classes, test_classes)

with open('merged_clean.json', 'w') as f:
    json.dump(merged, f)
"""