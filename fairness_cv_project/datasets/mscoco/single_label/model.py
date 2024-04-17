from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils
import numpy as np


class ObjectSingleLabel(nn.Module):
    def __init__(self, num_classes, device, params=None):
        super(ObjectSingleLabel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.base = models.resnet50()
        if params is not None:
            self.base.load_state_dict(torch.load(params))

        for param in self.base.parameters():
            param.requires_grad = False

        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, num_classes)
        self.base = self.base.to(self.device)

    def forward(self, x):
        return self.base(x)

    def state_dict(self):
        return self.base.state_dict()

    def classes(self):
        return self.num_classes

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        return self.base.load_state_dict(state_dict, strict)
