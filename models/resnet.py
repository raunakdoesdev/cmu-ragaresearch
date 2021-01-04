from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F

from src.training import Boilerplate


class Resnet50(Boilerplate):
    def test_visualizations(self, y_score, y_pred, y_true):
        pass

    def __init__(self, num_classes=100):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=[3, 7], stride=[1, 2], padding=[1, 3],  # only 1 input channel
                                     bias=False)
        self.model.fc = nn.Linear(512 * 4, num_classes)  # modify last layer of network

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = F.log_softmax(self.model(x), dim=1)
        return x
