from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch

from src.training import Boilerplate


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d([1, 2])
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_predict_features=300, num_out=30):
        super(Block, self).__init__()
        # intermediate1 = in_channels + (out_channels - in_channels) // 3
        # intermediate2 = in_channels + 2 * (out_channels - in_channels) // 3
        intermediate1 = (in_channels + out_channels) // 2
        self.sequential = nn.Sequential(OrderedDict([
            ('conv_block1', ConvBlock(in_channels, intermediate1)),
            ('conv_block2', ConvBlock(intermediate1, out_channels)),
            # ('conv_block3', ConvBlock(intermediate2, out_channels))
        ]))
        self.conv1x1 = nn.Conv2d(out_channels, num_predict_features, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d([1, 1])
        self.fc1 = nn.Linear(num_predict_features, num_out)

    def pred_block(self, x):
        x = self.conv1x1(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

    def forward(self, x, output_prob=False):
        if output_prob:
            return self.pred_block(self.sequential(x))
        else:
            return self.sequential(x)


class BlockedPhononet(Boilerplate):
    def __init__(self, num_out=30):
        super(BlockedPhononet, self).__init__()
        self.blocks = nn.Sequential(OrderedDict([
            ('block1', Block(1, 128, num_out=num_out)),
            ('block2', Block(128, 512, num_out=num_out)),
            ('block3', Block(512, 512, num_out=num_out))
        ]))

    def forward(self, x, num_layers=3, keepall=False):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        for i in range(num_layers - 1):
            x = self.blocks[i](x)

        return self.blocks[num_layers - 1](x, output_prob=True)

    def training_step(self, batch, batch_idx):
        loss = 0
        div = 0
        for i, single_batch in enumerate(batch):  # iterate through multiple chunk sizes
            if len(single_batch) != 2:   # check if empty
                continue
            x, y_true = single_batch
            try:
                y_score = self.forward(x, num_layers=i + 1)
                loss += F.nll_loss(y_score, y_true) * len(y_true)
                div += len(y_true)
            except Exception as e:
                import traceback
                print(traceback.format_exc())

        return {'loss': loss / div}


if __name__ == '__main__':
    model = BlockedPhononet()
    print(summary(model, (1, 12, 5000)))
