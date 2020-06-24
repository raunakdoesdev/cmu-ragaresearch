import copy
import multiprocessing as mp
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import metrics
from torch.autograd.function import once_differentiable
from torch.utils.data import DataLoader
import numpy as np


class PeriodicPadding2d(Function):
    @staticmethod
    def forward(ctx, input, pad):
        output = np.pad(input, pad, mode='wrap')  # find the function that performs what you want
        ctx.pad = pad
        ctx.size = input.size()
        ctx.numel = input.numel()
        return output

    @once_differentiable
    @staticmethod
    def backward(ctx, grad_output):
        pad = ctx.pad
        idx = grad_output.new(ctx.size)
        torch.arange(0, ctx.numel, out=idx)
        idx = np.pad(idx, pad, mode='wrap')  # or whatever is the function
        grad_input = grad_output.new(ctx.numel).zero_()
        grad_input.index_add_(0, idx, grad_output.view(-1))
        return grad_input.view(ctx.size)


class PhonoNet(pl.LightningModule):
    def __init__(self, train_set, val_set, hparams, dropout=0.5):
        super(PhonoNet, self).__init__()
        self.hparams = hparams
        self.train_set = train_set
        self.val_set = val_set
        self.encoder = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(1)),
            ('pad0', nn.ZeroPad2d((1, 1, 0, 0))),
            ('conv1', nn.Conv2d(1, 64, 3, padding=(1, 0), padding_mode='circular')),
            ('relu1', nn.LeakyReLU()),
            ('norm1', nn.BatchNorm2d(64)),
            ('pool1', nn.MaxPool2d([1, 2])),
            ('drop1', nn.Dropout(p=dropout)),
            ('pad1', nn.ZeroPad2d((1, 1, 0, 0))),
            ('conv2', nn.Conv2d(64, 128, 3, padding=(1, 0), padding_mode='circular')),
            ('relu2', nn.LeakyReLU()),
            ('norm2', nn.BatchNorm2d(128)),
            ('pool2', nn.MaxPool2d([1, 3])),
            ('drop2', nn.Dropout(p=dropout)),
            ('pad2', nn.ZeroPad2d((1, 1, 0, 0))),
            ('conv3', nn.Conv2d(128, 150, 3, padding=(1, 0), padding_mode='circular')),
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

        self.fc1 = nn.Linear(200, 30)

    def forward(self, x):
        x = x.unsqueeze(1)  # add empty channel dimension
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_out = self(x)
        ret = {'Cross Entropy Loss': F.cross_entropy(y_out, y_true)}

        y_score = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_score, 1)

        y_true = y_true.cpu()
        y_score = y_score.cpu()
        y_pred = y_pred.cpu()
        ret['Accuracy'] = torch.Tensor([metrics.accuracy_score(y_true, y_pred)])
        # ret['F1 Score'] = metrics.precision_score(y_true, y_pred, )
        # ret['PR AUC'] =  metrics.average_precision_score(y_true, y_score)

        return ret

    def validation_epoch_end(self, outputs):
        metrics = outputs[0].keys()
        log = {}
        for metric in metrics:
            log[f'val_{metric}'] = torch.stack([x[metric] for x in outputs]).mean()
        log['step'] = self.current_epoch
        log['log'] = copy.deepcopy(log)
        return log

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=mp.cpu_count(),
                          pin_memory=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=mp.cpu_count(),
                          pin_memory=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class RagaDetector:
    def __init__(self, batch_size=100, gpus=None):
        self.batch_size = batch_size
        self.gpus = gpus

    def fit(self, train, val):
        import torch.multiprocessing
        torch.multiprocessing.freeze_support()

        parser = ArgumentParser()
        parser.add_argument('--learning_rate', type=float, default=0.01)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        args = parser.parse_args()

        self.phono_net = PhonoNet(train, val, args)
        trainer = Trainer(gpus=self.gpus, logger=TensorBoardLogger('tb_logs'))

        # Find learning rate
        lr_finder = trainer.lr_find(self.phono_net)
        new_lr = lr_finder.suggestion()
        self.phono_net.hparams.lr = new_lr
        print(f"Optimal Learning Rate: {new_lr}")

        trainer.fit(self.phono_net)
