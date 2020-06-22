import copy
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from sklearn import metrics


class PhonoNet(pl.LightningModule):
    def __init__(self, train_set, val_set, hparams, dropout=0.15):
        super(PhonoNet, self).__init__()
        self.hparams = hparams
        self.train_set = train_set
        self.val_set = val_set
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 64, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.LeakyReLU()),
            ('pool1', nn.MaxPool2d([1, 2])),
            ('drop1', nn.Dropout(p=dropout)),

            ('conv2', nn.Conv2d(64, 128, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool2d([1, 3])),
            ('drop2', nn.Dropout(p=dropout)),

            ('conv3', nn.Conv2d(128, 150, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(150)),
            ('relu3', nn.LeakyReLU()),
            ('pool3', nn.MaxPool2d([4, 2])),
            ('drop3', nn.Dropout(p=dropout)),

            ('conv4', nn.Conv2d(150, 200, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(200)),
            ('gba', nn.AdaptiveAvgPool2d([1, 1])),
            ('drop4', nn.Dropout(p=dropout))
        ]))

        self.fc1 = nn.Linear(200, 40)

    def forward(self, x):
        x = x.unsqueeze(1)  # add empty channel dimension
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_score = self(x)
        _, y_pred = torch.max(y_score, 1)
        return {'Cross Entropy Loss': F.cross_entropy(y_score, y_true),
                'ROC AUC': metrics.roc_auc_score(y_true, y_score),
                'Accuracy': metrics.accuracy_score(y_true, y_pred),
                'F1 Score': metrics.precision_score(y_true, y_pred),
                'PR AUC': metrics.average_precision_score(y_true, y_score)}

    def validation_epoch_end(self, outputs):
        metrics = outputs[0].keys()
        log = {}
        for metric in metrics:
            log[f'val_{metric}'] = torch.stack([x[metric] for x in outputs]).mean()
        log['log'] = copy.deepcopy(log)
        return log

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class RagaDetector:
    def __init__(self, batch_size=100, gpus=None):
        self.batch_size = batch_size
        self.gpus = gpus

    def fit(self, train, val):
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
