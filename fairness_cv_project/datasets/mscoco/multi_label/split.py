import random, json, os

def train_val_test_split(data_dict, test_ratio, val_ratio, random_seed=42):
    random.seed(random_seed)

    train_set = []
    val_set = []
    test_set = []

    data = list(data_dict.items())
    random.shuffle(data)


    # calculate number of images for each set
    data_len = len(data)
    num_test = int(data_len * test_ratio)
    num_val = int(data_len * val_ratio)
    num_train = data_len - num_test - num_val

    test_set = data[:num_test]
    val_set = data[num_test : num_test + num_val]
    train_set = data[num_test + num_val :]

    return dict(train_set), dict(val_set), dict(test_set)


# change directory to the annotations directory
os.chdir(os.path.expanduser("~/CV-Fairness/data/datasets/mscoco"))

# load data dict
data_dict = json.load(open("metadata/multi_label/annotations.json", "r"))

# calculate splits
train_set, val_set, test_set = train_val_test_split(data_dict, test_ratio=0.2, val_ratio=0.2)

print(len(train_set), len(val_set), len(test_set))

# export train, val, test sets to json
with open("metadata/multi_label/train.json", "w") as outfile:
    json.dump(train_set, outfile, indent=4)

with open("metadata/multi_label/val.json", "w") as outfile:
    json.dump(val_set, outfile, indent=4)

with open("metadata/multi_label/test.json", "w") as outfile:
    json.dump(test_set, outfile, indent=4)
