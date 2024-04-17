import pandas as pd
import json, os, sys, random


# columns:
# 1. num_images
# 2. num_males
# 3. num_females
# 4. min_num_gender
# 5. cooccuring
def get_stats_dict(data_dict):
    filtered_stats_dict = {}
    for target, imgs in data_dict.items():
        filtered_stats_dict[target] = {}
        filtered_stats_dict[target]["num_images"] = len(imgs)
        filtered_stats_dict[target]["num_males"] = 0
        filtered_stats_dict[target]["num_females"] = 0
        filtered_stats_dict[target]["min_num_gender"] = 0
        filtered_stats_dict[target]["cooccuring"] = {}
        for img, img_data in imgs.items():
            if img_data["gender"] == "male":
                filtered_stats_dict[target]["num_males"] += 1
            else:
                filtered_stats_dict[target]["num_females"] += 1
            for other_target in img_data["other_targets"]:
                # don't count person as a target because not useful
                if other_target == "person":
                    continue

                if other_target in filtered_stats_dict[target]["cooccuring"]:
                    filtered_stats_dict[target]["cooccuring"][other_target] += 1
                else:
                    filtered_stats_dict[target]["cooccuring"][other_target] = 1
        filtered_stats_dict[target]["cooccuring"] = list(
            filtered_stats_dict[target]["cooccuring"].items()
        )
        filtered_stats_dict[target]["cooccuring"].sort(key=lambda x: x[1], reverse=True)

    # convert to dataframe
    df = pd.DataFrame.from_dict(filtered_stats_dict, orient="index")

    # get min_num_gender column
    df["min_num_gender"] = df[["num_males", "num_females"]].min(axis=1)

    return df.sort_values(by=["min_num_gender"], ascending=False)


def get_balanced_target_dict(data_dict, seed=42):
    random.seed(seed)
    balanced_target_dict = {}

    for target, imgs_dict in data_dict.items():
        # count male and female images
        male_image_ids = []
        female_image_ids = []
        for img_id, img_data in imgs_dict.items():
            if img_data["gender"] == "male":
                male_image_ids.append(img_id)
            else:
                female_image_ids.append(img_id)

        # get balanced subset of images
        balance_threshold = min(len(male_image_ids), len(female_image_ids))
        print("balance threshold: ", balance_threshold)
        male_image_ids = random.sample(male_image_ids, balance_threshold)
        female_image_ids = random.sample(female_image_ids, balance_threshold)

        balanced_imgs_dict = {}
        for img_id, img_data in imgs_dict.items():
            if img_id in male_image_ids or img_id in female_image_ids:
                balanced_imgs_dict[img_id] = img_data.copy()

        balanced_target_dict[target] = balanced_imgs_dict

    return balanced_target_dict


def get_target_id_mapping(data_dict):
    target_id_mapping = {}
    for i, target in enumerate(data_dict.keys()):
        target_id_mapping[target] = i

    return target_id_mapping


def assign_id_to_data(data_dict, target_id_mapping):
    for target, imgs_dict in data_dict.items():
        for img_id, img_data in imgs_dict.items():
            img_data["target_id"] = target_id_mapping[target]

    return data_dict
