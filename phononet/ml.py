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


class RagaDetector(pl.LightningModule):
    def __init__(self, train, val, dropout=0.15):
        super(RagaDetector, self).__init__()
        self.train = train
        self.val = val
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
        x = x.squeeze().unfold(1, 250, 250).permute(1, 0, 2).unsqueeze(1)  # convert song to chunks -> batches
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.repeat(y_hat.shape[0])
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        log = {'val_loss': avg_loss}
        log['log'] = copy.deepcopy(log)
        return log

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.repeat(y_hat.shape[0])
        return {'loss': F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class RagaDetector()
    def fit(self, train, val):
        parser = ArgumentParser()
        parser.add_argument('--learning_rate', type=float, default=0.01)
        args = parser.parse_args()

        raga_detector = RagaDetector(train, val)
        trainer = Trainer(gpus=4, logger=TensorBoardLogger('tb_logs'))

        # Find learning rate
        lr_finder = trainer.lr_find(raga_detector)
        new_lr = lr_finder.suggestion()
        raga_detector.hparams.lr = new_lr

        trainer.fit(raga_detector)
