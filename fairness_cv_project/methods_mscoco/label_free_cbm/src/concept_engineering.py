import os
import json
from pathlib import Path
import sys

sys.path.insert(1, str(Path.cwd()))

import openai

from fairness_cv_project.methods_mscoco.label_free_cbm.src.utils import data_utils

from fairness_cv_project.methods_mscoco.label_free_cbm.src.utils import conceptset_utils


def generate_concepts():
    dataset = "mscoco_single"
    prompt_types = ["important", "around", "superclass", "job"]

    openai.api_key = open(
        os.path.join(os.path.expanduser("~"), ".openai_api_key"), "r"
    ).read()[:-1]

    prompts = {
        "important": 'List the most important features for recognizing something as a "goldfish":\n\n-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\nList the most important features for recognizing something as a "beerglass":\n\n-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\nList the most important features for recognizing something as a "{}":',
        "superclass": 'Give superclasses for the word "tench":\n\n-fish\n-vertebrate\n-animal\n\nGive superclasses for the word "beer glass":\n\n-glass\n-container\n-object\n\nGive superclasses for the word "{}":',
        "around": 'List the things most commonly seen around a "tench":\n\n- a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\nList the things most commonly seen around a "beer glass":\n\n- beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\nList the things most commonly seen around a "{}":',
        "job": 'List the things commonly seen around a "teacher":\n\n- a classroom\n-a desk\n-a chalkboard\n-a book\n-a student\n-a pencil\n-a pen\n\nList the things commonly seen around a "software engineer":\n\n- a computer\n-a desk\n-a keyboard\n-a mouse\n-a monitor\n-a chair\n-a person\n\nList the things commonly seen around someone\'s working as a "{}":',
    }

    for prompt_type in prompt_types:
        base_prompt = prompts[prompt_type]

        cls_file = data_utils.LABEL_FILES[dataset]

        print(Path.cwd())

        with open(Path.cwd() / cls_file, "r") as f:
            classes = f.read().split("\n")

        feature_dict = {}

        for i, label in enumerate(classes):
            feature_dict[label] = set()
            print("\n", i, label)
            for _ in range(2):
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=base_prompt.format(label),
                    temperature=0.7,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                # clean up responses
                print(response)
                features = response["choices"][0]["text"]
                features = features.split("\n-")
                features = [feat.replace("\n", "") for feat in features]
                features = [feat.strip() for feat in features]
                features = [feat for feat in features if len(feat) > 0]
                features = set(features)
                feature_dict[label].update(features)
            feature_dict[label] = sorted(list(feature_dict[label]))

        json_object = json.dumps(feature_dict, indent=4)
        with open(
            Path.cwd()
            / "data/concept_sets/gpt3_init/gpt3_{}_{}.json".format(
                dataset, prompt_type
            ),
            "w",
        ) as outfile:
            outfile.write(json_object)


def filter_concepts():
    """
    CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
    OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
    MAX_LEN: max number of characters in a concept

    PRINT_PROB: what percentage of filtered concepts will be printed
    """

    CLASS_SIM_CUTOFF = 0.85
    OTHER_SIM_CUTOFF = 0.9
    MAX_LEN = 30
    PRINT_PROB = 1

    dataset = "mscoco_single"
    device = "cuda"

    root = Path.cwd()
    # CASE 1: You just generated concepts from GPT
    # EDIT these to use the initial concept sets you want

    save_name = "data/concept_sets/{}_filtered.txt".format(dataset)

    with open(
        root / "data/concept_sets/gpt3_init/gpt3_{}_important.json".format(dataset), "r"
    ) as f:
        important_dict = json.load(f)
    with open(
        root / "data/concept_sets/gpt3_init/gpt3_{}_superclass.json".format(dataset),
        "r",
    ) as f:
        superclass_dict = json.load(f)
    with open(
        root / "data/concept_sets/gpt3_init/gpt3_{}_around.json".format(dataset), "r"
    ) as f:
        around_dict = json.load(f)
    with open(
        root / "data/concept_sets/gpt3_init/gpt3_{}_job.json".format(dataset), "r"
    ) as f:
        job_dict = json.load(f)

    with open(root / data_utils.LABEL_FILES[dataset], "r") as f:
        classes = f.read().split("\n")

    concepts = set()

    for values in important_dict.values():
        concepts.update(set(values))

    for values in superclass_dict.values():
        concepts.update(set(values))

    for values in around_dict.values():
        concepts.update(set(values))

    for values in job_dict.values():
        concepts.update(set(values))

    print(len(concepts))
    """
    # CASE 2: Read another concept set, and filter it

    save_name = "data/concept_sets/{}_augmented_filtered.txt".format(dataset)

    with open(root / 'data' / 'concept_sets' / 'phoning_eating_augmented.txt') as f:
        concepts = set(f.read().split("\n"))
        
    with open(root / data_utils.LABEL_FILES[dataset], "r") as f:
        classes = f.read().split("\n")
        
    """
    concepts = conceptset_utils.remove_too_long(concepts, MAX_LEN, PRINT_PROB)

    concepts = conceptset_utils.filter_too_similar_to_cls(
        concepts, classes, CLASS_SIM_CUTOFF, device, PRINT_PROB
    )

    concepts = conceptset_utils.filter_too_similar(
        concepts, OTHER_SIM_CUTOFF, device, PRINT_PROB
    )

    with open(save_name, "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)


if __name__ == "__main__":
    filter_concepts()
    # generate_concepts()
