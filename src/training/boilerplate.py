import pytorch_lightning as pl
import pytorch_lightning.metrics
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import *
import copy

import attributedict.collections as coll

from src import chunk_chroma


class Boilerplate(pl.LightningModule):
    """
    Wrapper module which encapsulates all boiler plate code for training neural network (training/validation/testing).
    """

    def __init__(self):
        super(Boilerplate, self).__init__()
        self.hparams = coll.AttributeDict({'learning_rate': 0.01})

    def forward(self, x):
        raise NotImplementedError('Override me!')

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_score = self(x)
        _, y_pred = torch.max(y_score, 1)

        return {'y_score': y_score, 'y_pred': y_pred, 'y_true': y_true}

    def validation_epoch_end(self, outputs):
        # Get y_score / y_pred
        y_score = torch.cat([x['y_score'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        log = {'val_' + k: v for k, v in self.get_metrics(y_pred, y_true).items()}
        log['step'] = self.current_epoch
        log['log'] = copy.deepcopy(log)
        return log

    def get_metrics(self, y_pred, y_true):
        if not hasattr(self, 'accuracy'):
            self.accuracy = pl.metrics.sklearns.Accuracy()
        return {'accuracy': self.accuracy(y_pred, y_true),
                'f1': f1_score(y_pred, y_true)}

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        x = chunk_chroma(x.squeeze(), chunk_size=self.hparams.chunk_size)
        y_score = torch.mean(self(x), dim=0, keepdim=True)
        _, y_pred = torch.max(y_score, 1)
        return {'y_score': y_score, 'y_pred': y_pred, 'y_true': y_true}

    def test_visualizations(self, y_score, y_pred, y_true):
        raise NotImplementedError('Override this function with your own visualizations!')

    def test_epoch_end(self, outputs):
        # Get y_score / y_pred
        y_score = torch.cat([x['y_score'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        self.test_visualizations(y_score, y_pred, y_true)
        return {}

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_score = self(x)
        ret = {'loss': F.nll_loss(y_score, y_true)}
        _, y_pred = torch.max(y_score, 1)

        ret['y_score'] = y_score
        ret['y_pred'] = y_pred
        ret['y_true'] = y_true

        return ret

    def training_epoch_end(self, outputs):
        # Get y_score / y_pred
        y_score = torch.cat([x['y_score'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        log = {'train_' + k: v for k, v in self.get_metrics(y_pred, y_true).items()}
        log['step'] = self.current_epoch
        log['log'] = copy.deepcopy(log)
        return log

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
