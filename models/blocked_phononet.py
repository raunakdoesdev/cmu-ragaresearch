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
    def __init__(self, in_channels, out_channels, num_predict_features=300):
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
        self.fc1 = nn.Linear(num_predict_features, 30)

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
    def __init__(self, ):
        super(BlockedPhononet, self).__init__()
        self.blocks = nn.Sequential(OrderedDict([
            ('block1', Block(1, 128)),
            ('block2', Block(128, 512)),
            ('block3', Block(512, 512))
        ]))

    def forward(self, x, num_layers=3):
        for i in range(num_layers - 1):
            x = self.blocks[i](x)

        return self.blocks[num_layers - 1](x, output_prob=True)

    def training_step(self, batch, batch_idx):
        x, y_true = batch  # full song from data loader

        ret = {}
        # x will have shape 1, 12, 200 (batch size = 1)
        # y_score = self.forward(x.unsqueeze(0))
        loss = 0 # 3 * F.nll_loss(y_score, y_true)
        # _, y_pred = torch.max(y_score, 1)
        # ret['y_score'] = y_score
        # ret['y_pred'] = y_pred
        # ret['y_true'] = y_true

        count = 0
        for i, chunk_size in enumerate([75]):
            if chunk_size > x.shape[2]:
                continue
            count += 1
            unfolded = x.unfold(2, chunk_size, chunk_size).permute(2, 0, 1, 3)
            # print(unfolded.shape)
            y_score = self.forward(unfolded, num_layers=i + 1)
            loss += F.nll_loss(y_score, torch.cat(len(unfolded) * [y_true]))
        #
        # y_score = self(x)
        ret['loss'] = loss / (count + 3)
        #
        return ret


if __name__ == '__main__':
    phononet = BlockedPhononet()
    print(summary(phononet, (1, 12, 500)))
