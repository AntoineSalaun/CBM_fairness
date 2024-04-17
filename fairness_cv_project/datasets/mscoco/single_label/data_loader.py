import json, os

from PIL import Image

import torch
import torch.utils.data as data


class CocoLoader(data.Dataset):
    def __init__(self, image_dir, antn_file, transform=None, filter_func=None):
        self.image_dir = image_dir
        self.antn_file = antn_file
        self.transform = transform

        print("loading annotations into memory...")
        antns = json.load(open(antn_file))
        self.antn_data = list(antns.items())
        if filter_func is not None:
            self.antn_data = list(filter(filter_func, self.antn_data))
        self.targets = [img_data["target_id"] for img_id, img_data in self.antn_data]

        print(len(self.antn_data), "annotations loaded")

    def __len__(self):
        return len(self.antn_data)

    def __getitem__(self, index):
        img_id, img_data = self.antn_data[index]
        img_filename = img_data["file_name"]
        img_target = img_data["target_id"]
        img_path = os.path.join(self.image_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, img_target

    def classes(self):
        return range(len(set(self.targets)))
