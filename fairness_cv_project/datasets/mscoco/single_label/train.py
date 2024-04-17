import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from pathlib import Path
import random

sys.path.insert(1, str(Path.cwd()))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

from data_loader import CocoLoader
from model import ObjectSingleLabel


def main():
    LR = 0.01
    MOMENTUM = 0.9
    STEP_SIZE = 7
    GAMMA = 0.1
    NUM_EPOCHS = 25
    SEED = 0

    parser = argparse.ArgumentParser(description="Settings for training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="balanced",
        help="dataset name",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="balanced.pt",
        help="name of the saved model",
    )
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="momentum")
    parser.add_argument("--step_size", type=int, default=STEP_SIZE, help="step size")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="gamma")
    parser.add_argument(
        "--num_epochs", type=int, default=NUM_EPOCHS, help="number of epochs"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")

    random.seed(SEED)
    torch.manual_seed(SEED)

    args = parser.parse_args()

    path_root = Path.cwd() / "data" / "datasets" / "mscoco"
    path_dataset = (
        path_root / "data" / "mscoco" / "metadata" / "single_label" / args.dataset
    )
    path_save = Path(f"saved_models/mscoco/single_label/baseline/{args.save_name}")

    print(f"Training on {args.dataset}")

    img_size = 256
    crop_size = 224

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }

    antn_files = {
        "balanced": "data/datasets/mscoco/metadata/single_label/balanced_train.json",
        "imbalanced1": "data/datasets/mscoco/metadata/single_label/imbalanced1_train.json",
        "imbalanced2": "data/datasets/mscoco/metadata/single_label/imbalanced2_train.json",
    }

    images_path = "data/datasets/mscoco/images2014/"
    train_antn_file = antn_files[args.dataset]
    test_antn_file = "data/datasets/mscoco/metadata/single_label/val.json"

    image_datasets = {
        "train": CocoLoader(images_path, train_antn_file, data_transforms["train"]),
        "test": CocoLoader(images_path, test_antn_file, data_transforms["test"]),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size=16,
            shuffle=True,
            num_workers=8,
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"],
            batch_size=16,
            shuffle=True,
            num_workers=8,
        ),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    num_classes = len(image_datasets["train"].classes())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = ObjectSingleLabel(num_classes, device, "saved_models/resnet50.pt")

    model = models.resnet50()  # or another version of ResNet if you prefer
    model.load_state_dict(torch.load("saved_models/resnet50.pt"))

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_conv = optim.SGD(model.fc.parameters(), lr=LR, momentum=MOMENTUM)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=STEP_SIZE, gamma=GAMMA
    )

    model, best_acc, metadata = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        device,
        num_epochs=NUM_EPOCHS,
    )

    folders = str(path_dataset).split("/")
    name_dataset = "-".join(folders[-2:])

    if not os.path.exists(path_save.parent):
        os.makedirs(path_save.parent)
    torch.save(model.state_dict(), path_save)

    print(path_save.parent)

    with open(path_save.parent / "metadata_trainingp.txt", "w") as f:
        for line in metadata:
            f.write(f"{line}\n")
    with open(path_save.parent / "results.txt", "a") as f:
        f.write(f"{name_dataset}: {best_acc}\n")


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=25,
):
    metadata = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and testing phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to test mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for (
                inputs,
                labels,
            ) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            metadata.append(
                f"Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            )
            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, metadata


if __name__ == "__main__":
    main()
