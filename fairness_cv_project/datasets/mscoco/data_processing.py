import json
import random
import string
import os

# set random seed for reproducibility
random.seed(1)


def is_whole_word_in_text(word, text):
    # replace punctuation with spaces
    for p in string.punctuation:
        text = text.replace(p, " ")

    words = text.split()
    return word in words


def get_images_dict(filename):
    image_file = json.load(open(filename))
    images = image_file["images"]

    images_dict = {}
    for image in images:
        images_dict[image["id"]] = {"file_name": image["file_name"]}

    return images_dict


# read in captions from json file and associate them with their image id
def read_captions(filename, images_dict):
    caption_file = json.load(open(filename))
    captions = caption_file["annotations"]

    for caption in captions:
        image_id = caption["image_id"]
        caption_text = caption["caption"]
        if not "captions" in images_dict[image_id]:
            images_dict[image_id]["captions"] = []
        if not (caption_text in images_dict[image_id]["captions"]):
            images_dict[image_id]["captions"].append(caption_text.lower())


# classify images' gender; filtering out ambiguous images
def classify_images_gender(images_dict, male_labels, female_labels):
    images_dict_items = list(images_dict.items())
    for image_id, image_data in images_dict_items:
        male = False
        female = False
        for caption in image_data["captions"]:
            if any(is_whole_word_in_text(label, caption) for label in male_labels):
                male = True
            if any(is_whole_word_in_text(label, caption) for label in female_labels):
                female = True
        if male ^ female:
            if male:
                images_dict[image_id]["gender"] = "male"
            else:
                images_dict[image_id]["gender"] = "female"
        else:
            del images_dict[image_id]

    # remove unnecessary captions from images_dic
    for image_id, image_data in images_dict.items():
        del image_data["captions"]


# read in object targets from json file and associate them with their image id
# return a list of all categories
def read_targets(filename, images_dict):
    target_file = json.load(open(filename))
    cat_antns_raw = target_file["annotations"]
    categories = target_file["categories"]

    # remove unncessary information from category annotations
    cat_antns = []
    for antn in cat_antns_raw:
        new_antn = {
            "id": antn["id"],
            "image_id": antn["image_id"],
            "category_id": antn["category_id"],
        }
        cat_antns.append(new_antn)

    # create a category dictionary
    cat_dict = {}
    for cat in categories:
        cat_dict[cat["id"]] = cat["name"]

    # Associate categories with their names
    for antn in cat_antns:
        antn["category_name"] = cat_dict[antn["category_id"]]
        del antn["category_id"]

    # Associate categories with their images
    for antn in cat_antns:
        image_id = antn["image_id"]
        category_name = antn["category_name"]
        if image_id in images_dict:
            image_data = images_dict[image_id]
            if not "targets" in image_data:
                image_data["targets"] = []
            if not (category_name in image_data["targets"]):
                image_data["targets"].append(category_name)

    # Remove images that don't have a person target
    items = list(images_dict.items())
    for image_id, image_data in items:
        if (not "targets" in image_data) or (not "person" in image_data["targets"]):
            del images_dict[image_id]

    print(
        "number of images after removing images without a person target: ",
        len(images_dict),
    )

    return categories


# return dictionary with targets as keys and images as values
def categorize_images_by_targets(images_dict, categories):
    target_img_dict = {}
    for category in categories:
        category_name = category["name"]
        target_img_dict[category_name] = {}
        for image_id, image_data in images_dict.items():
            if not "targets" in image_data:
                continue
            if category_name in image_data["targets"]:
                target_img_dict[category_name][image_id] = image_data.copy()
                target_img_dict[category_name][image_id]["target"] = category_name
                target_img_dict[category_name][image_id]["other_targets"] = list(
                    filter(lambda x: x != category_name, image_data["targets"])
                )
                del target_img_dict[category_name][image_id]["targets"]

    return target_img_dict


# remove targets that have less than [threshold] images in the set, returning the filtered targets
def filter_irrelevant_images(target_img_dict, threshold=100):
    while True:
        irrelevant_targets = []
        for category_name, image_dict in target_img_dict.items():
            if len(target_img_dict[category_name]) < threshold:
                irrelevant_targets.append(category_name)

        # if there are no irrelevant targets, break
        if not irrelevant_targets:
            break

        # remove irrelevant targets from target_img_dict
        images_to_remove = set()
        for target in irrelevant_targets:
            images_to_remove.update(list(target_img_dict[target].keys()))
            del target_img_dict[target]
        print("images to remove: ", images_to_remove)
        print("number of images to remove: ", len(images_to_remove))
        for target, image_dict in target_img_dict.items():
            for image_id in images_to_remove:
                if image_id in image_dict:
                    del image_dict[image_id]

    return target_img_dict.keys()


def export_json(output_dict, filename):
    with open(filename, "w") as outfile:
        json.dump(output_dict, outfile, indent=4)


# process the dataset given captions and targets filenames
def process_data(male_labels, female_labels, captions_filename, targets_filename):
    print("processing data...")
    images_dict = get_images_dict(captions_filename)
    read_captions(captions_filename, images_dict)
    classify_images_gender(images_dict, male_labels, female_labels)
    print("number of images: ", len(images_dict))
    categories = read_targets(targets_filename, images_dict)
    target_img_dict = categorize_images_by_targets(images_dict, categories)
    return target_img_dict


# merge train and val dictionaries and return the merged dictionary
def merge_datasets(train_dict, val_dict):
    merged_dict = train_dict.copy()
    for target, image_dict in val_dict.items():
        if target in merged_dict:
            merged_dict[target].update(image_dict)
        else:
            merged_dict[target] = image_dict
    return merged_dict


# convert target-image dictionary to image annotations dictionary
def get_annotations(target_img_dict):
    annotations = {}
    for target, image_dict in target_img_dict.items():
        for img_id, img_data in image_dict.items():
            if not img_id in annotations:
                annotations[img_id] = img_data.copy()

    print("number of annotations: ", len(annotations))

    return annotations


# generate the full dataset as json file from the train and val dictionaries
def generate_full_dataset(
    male_labels,
    female_labels,
    train_captions_filename,
    train_targets_filename,
    val_captions_filename,
    val_targets_filename,
    output_filename,
    output_annotations_filename,
    filter_targets=True,
):
    train_dict = process_data(
        male_labels,
        female_labels,
        train_captions_filename,
        train_targets_filename,
    )
    val_dict = process_data(
        male_labels,
        female_labels,
        val_captions_filename,
        val_targets_filename,
    )
    merged_dict = merge_datasets(train_dict, val_dict)
    if filter_targets:
        remaining_targets = filter_irrelevant_images(merged_dict)
        print("remaining targets: ", remaining_targets)
    export_json(merged_dict, output_filename)
    annotations = get_annotations(merged_dict)
    export_json(annotations, output_annotations_filename)


# define gender labels
male_labels = ["man", "male"]
female_labels = ["woman", "female"]

# change directory to the data directory
os.chdir(os.path.expanduser("~/CV-Fairness/data/datasets/mscoco/"))

# define train filenames
train_captions_filename = "annotations/captions_train2014.json"
train_targets_filename = "annotations/instances_train2014.json"

# define val filenames
val_captions_filename = "annotations/captions_val2014.json"
val_targets_filename = "annotations/instances_val2014.json"

# run pipeline
generate_full_dataset(
    male_labels,
    female_labels,
    train_captions_filename,
    train_targets_filename,
    val_captions_filename,
    val_targets_filename,
    "full.json",
    "full_annotations.json",
    filter_targets=False,
)
