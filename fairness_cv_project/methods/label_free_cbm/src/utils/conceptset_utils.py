import math
from tqdm import tqdm
import random
import json
import os
import time

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

from fairness_cv_project.methods.label_free_cbm.src.clip import clip
import openai
    

# Hard coding of the path to all-mpnet-base-v2 model
MODEL_PATH = Path.cwd() / 'saved_models/all-mpnet-base-v2' 

openai.api_key = open(os.path.join(os.path.expanduser("~"), ".openai_api_key"), "r").read()[:-1]


def get_init_conceptnet(classes, limit=200, relations=["HasA", "IsA", "PartOf", "HasProperty", "MadeOf", "AtLocation"]):
    concepts = set()

    for cls in tqdm(classes):
        words = cls.replace(',', '').split(' ')
        for word in words:
            obj = requests.get('http://api.conceptnet.io/c/en/{}?limit={}'.format(word, limit)).json()
            obj.keys()
            for dicti in obj['edges']:
                rel = dicti['rel']['label']
                try:
                    if dicti['start']['language'] != 'en' or dicti['end']['language'] != 'en':
                        continue
                except(KeyError):
                    continue

                if rel in relations:
                    if rel in ["IsA"]: 
                        concepts.add(dicti['end']['label'])
                    else:
                        concepts.add(dicti['start']['label'])
                        concepts.add(dicti['end']['label'])
    return concepts


def remove_too_long(concepts, max_len, print_prob=0):
    """
    deletes all concepts longer than max_len
    """
    new_concepts = []
    for concept in concepts:
        if len(concept) <= max_len:
            new_concepts.append(concept)
        else:
            if random.random()<print_prob:
                print(len(concept), concept)
    # print(len(concepts), len(new_concepts))
    print(new_concepts)
    return new_concepts


def filter_too_similar_to_cls(concepts, classes, sim_cutoff, device="cuda", print_prob=0):
    """
    filters out concepts that are too similar to classes
    """
    #first check simple text matches
    # print(len(concepts))
    concepts = list(concepts)
    concepts = sorted(concepts)
    
    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix+cls)
                if random.random()<print_prob:
                    print("Class:{} - Deleting {}".format(cls, prefix+cls))
            except(ValueError):
                pass
        try:
            concepts.remove(cls.upper())
        except(ValueError):
            pass
        try:
            concepts.remove(cls[0].upper()+cls[1:])
        except(ValueError):
            pass
    # print(len(concepts))
        
    mpnet_model = SentenceTransformer(MODEL_PATH)
    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    dot_prods_c = _clip_dot_prods(classes, concepts)
    #weighted since mpnet has highger variance
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if j not in to_delete:
                    to_delete.append(j)
                    if random.random()<print_prob:
                        print("Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(classes[i], concepts[j], dot_prods[i,j], concepts[j]))
                        print("".format(concepts[j]))
                        
    to_delete = sorted(to_delete)[::-1]

    for item in to_delete:
        concepts.pop(item)
    # print(len(concepts))
    return concepts

def filter_too_similar(concepts, sim_cutoff, device="cuda", print_prob=0):
    
    mpnet_model = SentenceTransformer(MODEL_PATH)
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = concept_features @ concept_features.T
    dot_prods_c = _clip_dot_prods(concepts, concepts)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if i not in to_delete and j not in to_delete:
                    to_print = random.random() < print_prob
                    #Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                        to_delete.append(i)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[i]))
                    else:
                        to_delete.append(j)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[j]))
                            
    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def _clip_dot_prods(list1, list2, device="cuda", clip_name="ViT-B/16", batch_size=500):
    "Returns: numpy array with dot products"
    clip_model, _ = clip.load(clip_name, device=device)
    text1 = clip.tokenize(list1).to(device)
    text2 = clip.tokenize(list2).to(device)
    
    features1 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text1)/batch_size)):
            features1.append(clip_model.encode_text(text1[batch_size*i:batch_size*(i+1)]))
        features1 = torch.cat(features1, dim=0)
        features1 /= features1.norm(dim=1, keepdim=True)

    features2 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text2)/batch_size)):
            features2.append(clip_model.encode_text(text2[batch_size*i:batch_size*(i+1)]))
        features2 = torch.cat(features2, dim=0)
        features2 /= features2.norm(dim=1, keepdim=True)
        
    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()

def most_similar_concepts(word, concepts, device="cuda"):
    """
    returns most similar words to a given concepts
    """
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    word_features = mpnet_model.encode([word])
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = word_features @ concept_features.T
    dot_prods_c = _clip_dot_prods([word], concepts, device)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    min_distance, indices = torch.topk(torch.FloatTensor(dot_prods[0]), k=5)
    return [(concepts[indices[i]], min_distance[i]) for i in range(len(min_distance))]

def get_visual_scores(concepts, dataset):
    """
    get visual scores for each concept based on GPT's visualization score
    """
    
    concepts = list(concepts)
    prompt_types = ['visualization', "scale", 'score', 'detectable']

    # prompts tried for prompt-engineering; "detectable" used for the study
    prompts = {
        "visualization" : "Give me a visualization score for \n\"monitor\":9/10\n\"keyboard\":10/10\n\"action\":2/10\n\"smile\":7/10\n\"love\":4/10\n\"a wind\":5/10\n\"a mental state\":1/10\n\"a handle on one side\":6/10\n\nGive me a visualization score for these words:",
        "scale" : "In the scale of 1 to 10, how visible is \n\"monitor\":9/10\n\"keyboard\":10/10\n\"action\":2/10\n\"smile\":7/10\n\"love\":4/10\n\"a wind\":5/10\n\"a mental state\":1/10\n\"a handle on one side\":6/10\n\nIn the scale of 1 to 10, how visible are these words:",
        "score" : "Can you score (in scale of 1 to 10) sif you can see \n\"monitor\":9/10\n\"keyboard\":10/10\n\"action\":2/10\n\"smile\":7/10\n\"love\":4/10\n\"a wind\":5/10\n\"a mental state\":1/10\n\"a handle on one side\":6/10\n\nCan you score (in scale of 1 to 10) if you can see these words:",
        "detectable" : "For human eyes, when looking at a picture, how detectable is (in scale of 1 to 10) \n\"monitor\":9/10\n\"keyboard\":10/10\n\"action\":2/10\n\"smile\":7/10\n\"love\":4/10\n\"a wind\":5/10\n\"a mental state\":1/10\n\"a handle on one side\":6/10\n\nFor human eyes, when looking at a picture, how detectable (in scale of 1 to 10) are these words:",
    }

    while concepts:
        # divide the concepts into groups of 15 to avoid overloading the language model with too many requests in a prompt
        sample_set = concepts[:15]

        for j in range(2):
            answer = ""
            prompt = prompts["detectable"]

            for i, label in enumerate(sample_set):
                prompt += "\n\"{}\":".format(label)

            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=prompt,
                temperature=0.7,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # helps with concurrency issues between the api calls
            time.sleep(1) 

            if j == 1:
                answer += response["choices"][0]["text"] + "\n"

                # does require manual processing/checking afterwards
                save_name = "data/concept_sets/visual_scores/visual_scores_{}_filtered.txt".format(dataset)
                with open(save_name, "a") as f:
                    f.write(answer)

                print(prompt)
                print("answer")
                print(answer)
                concepts = concepts[15:]   

def filter_by_visual_scores(scores, dataset):
    """
    From visual scores of the concepts in the dataset, create concept 
    files that filter based on concepts >= score per scores
    """     
    root = Path.cwd()
    concept_to_score = dict()

    # read the scores from the visual score file
    with open(root / "data/concept_sets/visual_scores/visual_scores_{}_filtered.txt".format(dataset), "r") as f:
        lines = f.read().split("\n")

    # process the files to create a dict of concepts to scores
    for line in lines:
        x = line.index("/10")
        score = 10 if line[x-2:x] == "10" else int(line[x-1])
        new_line = line[1:] if line[0] == "\"" else line
        y = new_line.index(":")
        concept = new_line[:y-1] if new_line[y-1] == "\"" else new_line[:y-1]
        concept_to_score[concept] = score

    # iterate through the scores to filter and make concept files accordingly
    for score in scores:
        save_name = "data/concept_sets/visual_concept_sets/visual_{}_{}_filtered.txt".format(score, dataset)
        filtered_concept_to_score = {k:v for k,v in concept_to_score.items() if v >= score}
        filtered_concepts = list(filtered_concept_to_score.keys())
        with open(save_name, "w") as f:
            f.write(filtered_concepts[0])
            for concept in filtered_concepts[1:]:
                f.write("\n" + concept)
