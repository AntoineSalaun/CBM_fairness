import random, json, os


# balance modes: balanced, imbalanced_1 (first half male), imbalanced_2 (first half female)
def train_val_test_split(data_dict, test_ratio, val_ratio, random_seed=42):
    random.seed(random_seed)

    balanced_train_set = []
    imbalanced1_train_set = []
    imbalanced2_train_set = []
    val_set = []
    test_set = []

    for idx, (target, imgs) in enumerate(data_dict.items()):
        male_imgs = []
        female_imgs = []
        for img_id, img_data in imgs.items():
            if img_data["gender"] == "male":
                male_imgs.append((img_id, img_data))
            else:
                female_imgs.append((img_id, img_data))

        random.shuffle(male_imgs)
        random.shuffle(female_imgs)

        # calculate number of images for each set
        imgs_len = len(male_imgs)
        num_test = int(imgs_len * test_ratio)
        num_val = int(imgs_len * val_ratio)
        num_train = imgs_len - num_test - num_val
        num_train_male_balanced = num_train // 2
        num_train_female_balanced = num_train // 2

        num_train_male_imbalanced1 = num_train if idx < len(data_dict) / 2 else 0
        num_train_female_imbalanced1 = 0 if idx < len(data_dict) / 2 else num_train

        num_train_male_imbalanced2 = 0 if idx < len(data_dict) / 2 else num_train
        num_train_female_imbalanced2 = num_train if idx < len(data_dict) / 2 else 0

        # add images to sets
        test_set.extend(male_imgs[:num_test])
        test_set.extend(female_imgs[:num_test])

        val_set.extend(male_imgs[num_test : num_test + num_val])
        val_set.extend(female_imgs[num_test : num_test + num_val])

        balanced_train_set.extend(male_imgs[num_test + num_val : num_test + num_val + num_train_male_balanced])
        balanced_train_set.extend(female_imgs[num_test + num_val : num_test + num_val + num_train_female_balanced])

        imbalanced1_train_set.extend(male_imgs[num_test + num_val : num_test + num_val + num_train_male_imbalanced1])
        imbalanced1_train_set.extend(female_imgs[num_test + num_val : num_test + num_val + num_train_female_imbalanced1])

        imbalanced2_train_set.extend(male_imgs[num_test + num_val : num_test + num_val + num_train_male_imbalanced2])
        imbalanced2_train_set.extend(female_imgs[num_test + num_val : num_test + num_val + num_train_female_imbalanced2])

    return (
        dict(balanced_train_set),
        dict(imbalanced1_train_set),
        dict(imbalanced2_train_set),
        dict(val_set),
        dict(test_set),
    )


# change directory to the annotations directory
os.chdir(os.path.expanduser("~/CV-Fairness/data/datasets/mscoco"))

# load data dict
data_dict = json.load(open("metadata/single_label/balanced.json", "r"))

# calculate splits
(
    balanced_train_set,
    imbalanced1_train_set,
    imbalanced2_train_set,
    val_set,
    test_set,
) = train_val_test_split(data_dict, test_ratio=0.2, val_ratio=0.2)

print(
    len(balanced_train_set),
    len(imbalanced1_train_set),
    len(imbalanced2_train_set),
    len(val_set),
    len(test_set),
)

# export train, val, test sets to json
with open("metadata/single_label/balanced_train.json", "w") as outfile:
    json.dump(balanced_train_set, outfile, indent=4)

with open("metadata/single_label/imbalanced1_train.json", "w") as outfile:
    json.dump(imbalanced1_train_set, outfile, indent=4)

with open("metadata/single_label/imbalanced2_train.json", "w") as outfile:
    json.dump(imbalanced2_train_set, outfile, indent=4)

with open("metadata/single_label/val.json", "w") as outfile:
    json.dump(val_set, outfile, indent=4)

with open("metadata/single_label/test.json", "w") as outfile:
    json.dump(test_set, outfile, indent=4)
