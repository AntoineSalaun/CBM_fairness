import json, os, sys, random
import pandas as pd

from collections import Counter
from itertools import chain

sys.path.append("../")

from data_utils import (
    get_stats_dict,
    get_balanced_target_dict,
    get_target_id_mapping,
    assign_id_to_data,
)


# get symmetric difference of a list of sets i.e. elements with frequency 1
def get_sym_diff(sets):
    freq = Counter(chain.from_iterable(sets))
    res = [k for k, v in freq.items() if v == 1]
    return res


def filter_data_dict(selected_targets, selected_images, data_dict):
    filtered_data_dict = {}
    count = 0
    for target in selected_targets:
        filtered_data_dict[target] = {}
        for img_id, img_data in data_dict[target].items():
            if img_id in selected_images:
                assert not any(
                    other_target in selected_targets
                    for other_target in img_data["other_targets"]
                )
                filtered_data_dict[target][img_id] = img_data
                count += 1
    print("number of images: ", count)
    return filtered_data_dict


def build_target_img_sets(data_dict):
    # Restructure data_dict into targets and their associated images
    data_list = []

    for target, imgs in data_dict.items():
        # discard person target at this stage
        if target == "person":
            continue
        data_list.append((target, set(imgs.keys())))
    data_list.sort(key=lambda x: len(x[1]), reverse=True)
    return data_list


def top_10_balanced_filter(data_dict):
    df = get_stats_dict(data_dict)
    top_10_targets_balanced = (
        df.sort_values(by=["min_num_gender"], ascending=False).head(10).index.tolist()
    )
    return top_10_targets_balanced


# filter images so that each image is only mapped to one target
def filter_single_label_imgs(data_dict, target_filter=None):
    data_list = build_target_img_sets(data_dict)

    # filter images
    selected_image_sets = []
    saved_imgs_len = 0

    for target, img_set in data_list:
        sym_diff = get_sym_diff([*([p[1] for p in selected_image_sets]), img_set])
        if len(sym_diff) > saved_imgs_len:
            saved_imgs_len = len(sym_diff)
            selected_image_sets.append((target, img_set))

    selected_images = get_sym_diff(([p[1] for p in selected_image_sets]))
    print("selected images", saved_imgs_len)
    selected_targets = [p[0] for p in selected_image_sets]
    print("selected targets", len(selected_targets), selected_targets)
    filtered_data_dict = filter_data_dict(selected_targets, selected_images, data_dict)

    if target_filter:
        # apply target filter
        filtered_targets = target_filter(filtered_data_dict)
        filtered_images = get_sym_diff(
            ([p[1] for p in selected_image_sets if p[0] in filtered_targets])
        )
        return filter_data_dict(filtered_targets, filtered_images, data_dict)
    else:
        # return filtered data_dict
        return filtered_data_dict


def pipeline(data_dict):
    filtered_data_dict = filter_single_label_imgs(
        data_dict, target_filter=top_10_balanced_filter
    )
    balanced_data_dict = get_balanced_target_dict(filtered_data_dict)
    target_id_mapping = get_target_id_mapping(balanced_data_dict)
    final_data_dict = assign_id_to_data(balanced_data_dict, target_id_mapping)
    return final_data_dict, target_id_mapping


# change directory to the annotations directory
os.chdir(os.path.expanduser("~/CV-Fairness/data/datasets/mscoco"))

data_dict = json.load(open("metadata/single_label/full.json", "r"))
final_data_dict, target_id_mapping = pipeline(data_dict)

# export them to json
with open("metadata/single_label/balanced.json", "w") as outfile:
    json.dump(final_data_dict, outfile, indent=4)
with open("metadata/single_label/target_to_id.json", "w") as outfile:
    json.dump(target_id_mapping, outfile, indent=4)
