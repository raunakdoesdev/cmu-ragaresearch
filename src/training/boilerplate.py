import copy

import attributedict.collections as coll
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adadelta
from pytorch_lightning.metrics.functional import *

from src import chunk_chroma


class Boilerplate(pl.LightningModule):
    """
    Wrapper module which encapsulates all boiler plate code for training neural network (training/validation/testing).
    """

    def __init__(self):
        super(Boilerplate, self).__init__()
        self.hparams = coll.AttributeDict({'lr': 0.01})
        self.epochs = 100

    def forward(self, x):
        raise NotImplementedError('Override me!')

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        # x = x.unsqueeze(0)
        y_score = self(x)
        _, y_pred = torch.max(y_score, 1)

        return {'y_score': y_score, 'y_pred': y_pred, 'y_true': y_true}

    def validation_epoch_end(self, outputs):
        # Get y_score / y_pred
        y_score = torch.cat([x['y_score'] for x in outputs])

        y_pred = torch.cat([x['y_pred'] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        # log = {'val_' + k: v for k, v in self.get_metrics(y_score, y_true).items()}
        log = self.get_val_metrics(y_score, y_true)
        log['step'] = self.current_epoch
        log['log'] = copy.deepcopy(log)
        return log

    def get_val_metrics(self, y_score, y_true):
        log = {'val_' + k: v for k, v in self.get_metrics(y_score, y_true).items()}
        return log

    def get_metrics(self, y_score, y_true):
        _, y_pred = torch.max(y_score, 1)
        accuracies = self.accuracy(y_score, y_true, (1, 3, 5))
        return {'accuracy': accuracies[0],
                'top3-accuracy': accuracies[1],
                'top5-accuracy': accuracies[2],
                'f1': f1_score(y_pred, y_true)}

    def accuracy(self, output, target, topk=(1,)):
        """
        Computes the accuracy over the k top predictions for the specified values of k
        output and targets are just tensors.
        """

        maxk = max(topk)
        batch_size = target.size(0)
        print(output.shape)
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0) / batch_size for k in topk]

    # def test_step(self, batch, batch_idx):
    #     x, y_true = batch
    #     x = chunk_chroma(x.squeeze(), chunk_size=self.hparams.chunk_size)
    #     y_score = torch.mean(self(x), dim=0, keepdim=True)
    #     _, y_pred = torch.max(y_score, 1)
    #     return {'y_score': y_score, 'y_pred': y_pred, 'y_true': y_true}

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_score = self(x)
        _, y_pred = torch.max(y_score, 1)
        return {'y_score': y_score, 'y_pred': y_pred, 'y_true': y_true}

    def aggregation_fn(self, x, y_true):
        return torch.mean(x, dim=0, keepdim=True)

    def test_visualizations(self, y_score, y_pred, y_true):
        raise NotImplementedError('Override this function with your own visualizations!')

    def test_epoch_end(self, outputs):
        # Get y_score / y_pred
        y_score = torch.cat([x['y_score'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        self.test_visualizations(y_score, y_pred, y_true)
        log = self.get_val_metrics(y_score, y_true)
        log['log'] = copy.deepcopy(log)
        return log

    def training_step(self, batch, batch_idx):
        x, y_true = batch  # full song from data loader
        y_score = self(x)
        ret = {'loss': F.nll_loss(y_score, y_true)}
        _, y_pred = torch.max(y_score, 1)

        ret['y_score'] = y_score
        ret['y_pred'] = y_pred
        ret['y_true'] = y_true

        return ret

    def configure_optimizers(self):
        print(f'{self.epochs} = planned # of epochs')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return [optimizer], [scheduler]
