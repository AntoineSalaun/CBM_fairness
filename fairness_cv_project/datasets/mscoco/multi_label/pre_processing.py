import json, os, sys, random
import pandas as pd

sys.path.append("../")

from data_utils import (
    get_balanced_target_dict,
    get_target_id_mapping,
    assign_id_to_data,
)


def remove_person_target(data_dict):
    # remove person target from data_dict
    del data_dict["person"]
    for target, imgs in data_dict.items():
        for img_id, img_dict in imgs.items():
            if "person" in img_dict["other_targets"]:
                img_dict["other_targets"].remove("person")
    return data_dict


def flatten_data_dict(data_dict):
    # flatten the data_dict
    flattened_data_dict = {}
    for key, imgs in data_dict.items():
        for img_id, img_dict in imgs.items():
            if img_id not in flattened_data_dict:
                flattened_data_dict[img_id] = img_dict.copy()
                flattened_data_dict[img_id]["targets"] = [img_dict["target"]] + img_dict["other_targets"]
                del flattened_data_dict[img_id]["target"]
                del flattened_data_dict[img_id]["other_targets"]
    return flattened_data_dict


def assign_ids_to_data(data_dict, target_id_mapping):
    for img_id, img_dict in data_dict.items():
        img_dict["target_ids"] = [
            target_id_mapping[target] for target in img_dict["targets"]
        ]

    return data_dict


def pipeline(data_dict):
    filtered_data_dict = remove_person_target(data_dict)
    # balanced_data_dict = get_balanced_target_dict(filtered_data_dict)
    annotations = flatten_data_dict(filtered_data_dict)
    target_id_mapping = get_target_id_mapping(filtered_data_dict)
    final_data_dict = assign_ids_to_data(annotations, target_id_mapping)
    return final_data_dict, target_id_mapping


# change directory to the annotations directory
os.chdir(os.path.expanduser("~/CV-Fairness/data/datasets/mscoco"))

data_dict = json.load(open("metadata/full.json", "r"))
final_data_dict, target_id_mapping = pipeline(data_dict)

# export them to json
with open("metadata/multi_label/annotations.json", "w") as outfile:
    json.dump(final_data_dict, outfile, indent=4)
with open("metadata/multi_label/target_to_id.json", "w") as outfile:
    json.dump(target_id_mapping, outfile, indent=4)
