from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from src.training import Boilerplate


class Phononet(Boilerplate):
    def __init__(self, dropout=0.1):
        super(Phononet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('init_norm', nn.BatchNorm2d(1)),
            ('only_time_conv', nn.Conv2d(1, 100, [1, 100], stride=[1, 5])),
            ('relu0', nn.LeakyReLU()),
            ('maxpool', nn.MaxPool2d([1, 2])),
            ('reduce_time_conv', nn.Conv2d(100, 1, 1)),
            ('relu0p5', nn.LeakyReLU()),

            ('norm0', nn.BatchNorm2d(1)),
            ('conv1', nn.Conv2d(1, 64, 3, padding=1)),
            ('relu1', nn.LeakyReLU()),
            ('norm1', nn.BatchNorm2d(64)),
            ('pool1', nn.MaxPool2d([1, 2])),
            ('drop1', nn.Dropout(p=dropout)),

            ('conv2', nn.Conv2d(64, 128, 3, padding=1)),
            ('relu2', nn.LeakyReLU()),
            ('norm2', nn.BatchNorm2d(128)),
            ('pool2', nn.MaxPool2d([1, 3])),
            ('drop2', nn.Dropout(p=dropout)),

            ('conv3', nn.Conv2d(128, 150, 3, padding=1)),
            ('relu3', nn.LeakyReLU()),
            ('norm3', nn.BatchNorm2d(150)),
            ('pool3', nn.MaxPool2d([4, 2])),
            ('drop3', nn.Dropout(p=dropout)),

            ('conv4', nn.Conv2d(150, 200, 3, padding=1)),
            ('relu4', nn.LeakyReLU()),
            ('norm4', nn.BatchNorm2d(200)),
            ('gba', nn.AdaptiveAvgPool2d([1, 1])),
            ('drop4', nn.Dropout(p=dropout))
        ]))

        self.fc1 = nn.Linear(200, 40)

    def forward(self, x):
        x = x.unsqueeze(1)  # add empty channel dim
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)  # flatten
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x
