import copy
import multiprocessing as mp
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import metrics
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from pytorch_lightning.metrics.functional import *
from tqdm import tqdm
import seaborn as sns
from phononet import transpose_chromagram, plotter, ImbalancedDatasetSampler
import numpy as np
import plotly.express as px


class PhonoNetNetwork(nn.Module):
    def __init__(self, dropout=0.1):
        super(PhonoNetNetwork, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
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
        """
        :param x: tensor of shape (-1, 1, 12, 5000)
        :return: resultant tensor after operations
        """
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)
        return x

    def generate_embedding(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)  # Flatten
        return x


class PhonoNet(pl.LightningModule):
    def __init__(self, network, train_set, val_set, hparams):
        super(PhonoNet, self).__init__()
        self.hparams = hparams
        self.train_set = train_set
        self.val_set = val_set
        self.network = network

    def forward(self, x):
        x = x.unsqueeze(1)  # add empty channel dimension
        return self.network(x)

    def generate_embedding(self, x):
        x = x.unsqueeze(1)
        return self.network.generate_embedding(x)

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_out = self(x)

        y_score = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_score, 1)

        return {'y_score': y_score, 'y_pred': y_pred, 'y_true': y_true}

    def validation_epoch_end(self, outputs):
        # Get y_score / y_pred
        y_score = torch.cat([x['y_score'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        # Metrics:
        log = {
            'val_accuracy': accuracy(y_pred, y_true),
            'val_f1': f1_score(y_pred, y_true),
            # 'Confusion Matrix': confusion_matrix(y_pred, y_true)
        }

        pass

        log['step'] = self.current_epoch
        log['log'] = copy.deepcopy(log)
        return log

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=mp.cpu_count(),
                          pin_memory=True)

    def test_step(self, batch, batch_idx):
        x, y_true = batch

        y_out = self(x)

        y_score = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_score, 1)

        y_score = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_score, 1)

        return {'y_score': y_score, 'y_pred': y_pred, 'y_true': y_true}

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=72)

    def test_epoch_end(self, outputs):
        # Get y_score / y_pred
        y_score = torch.cat([x['y_score'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        # Plot Accuracy vs. Class
        cm = confusion_matrix(y_pred, y_true).cpu().numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plotter.plot_class_accuracies(100 * cm.diagonal())

        # Plot F1 vs. Class
        f1 = pl.metrics.sklearns.F1(average=None)
        f1_scores = f1(y_true.cpu(), y_pred.cpu())
        plotter.plot_class_f1(f1_scores)

        # Plot Confusion Matrix
        cm = confusion_matrix(y_pred, y_true).cpu().numpy()
        plotter.plot_confusion_matrix(cm)

        plotter.save_figs('test.pdf')

        return {'log': {}}

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_out = self(x)

        ret = {'loss': F.cross_entropy(y_out, y_true)}

        y_score = F.softmax(y_out, dim=1)
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

        # Metrics:
        log = {
            'train_accuracy': accuracy(y_pred, y_true),
            'train_f1': f1_score(y_pred, y_true),
            # 'Confusion Matrix': confusion_matrix(y_pred, y_true)
        }

        pass

        log['step'] = self.current_epoch
        log['log'] = copy.deepcopy(log)
        return log

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=mp.cpu_count(),
                          pin_memory=True, sampler=ImbalancedDatasetSampler(self.train_set))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class RagaTrainer:
    def __init__(self, max_epochs=100, batch_size=100, gpus=None):
        self.batch_size = batch_size
        self.gpus = gpus
        self.max_epoch = max_epochs

    def fit(self, network, train, val):
        import torch.multiprocessing
        torch.multiprocessing.freeze_support()

        parser = ArgumentParser()
        parser.add_argument('--learning_rate', type=float, default=0.01)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        hparams = parser.parse_args()

        self.phono_net = PhonoNet(network, train, val, hparams)
        tb = TensorBoardLogger('tb_logs')

        checkpoint_callback = ModelCheckpoint(
            monitor='val_accuracy',
            mode='max'
        )

        trainer = Trainer(gpus=self.gpus, logger=tb, max_epochs=self.max_epoch,
                          checkpoint_callback=checkpoint_callback)

        # Find learning rate
        lr_finder = trainer.lr_find(self.phono_net)
        new_lr = lr_finder.suggestion()
        self.phono_net.hparams.lr = new_lr
        print(f"Optimal Learning Rate: {new_lr}")

        trainer.fit(self.phono_net)

    def evaluate(self, network, train, val, path='/home/raunak/cmu-ragaresearch/main_scripts/default/version_55/checkpoints/epoch=69.ckpt'):
        parser = ArgumentParser()
        parser.add_argument('--learning_rate', type=float, default=0.01)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        hparams = parser.parse_args()

        self.phono_net = PhonoNet(network, train, val, hparams)
        self.phono_net.load_state_dict(torch.load(path)['state_dict'])
        trainer = Trainer(gpus=self.gpus, logger=None, max_epochs=self.max_epoch)
        trainer.test(self.phono_net)

    def tsne(self, network, dataset):
        import pickle
        labels, embeddings = pickle.load(open('tsne.pkl', 'rb'))
        tsne_embeddings = TSNE(perplexity=50).fit_transform(embeddings)
        ragas = ['Abhogi', 'Ahir bhairav', 'Alahiya bilawal', 'Bageshree', 'Bairagi', 'Basant', 'Bhairav', 'Bhoop',
                 'Bihag', 'Bilaskhani todi', 'Darbari', 'Des', 'Gaud malhar', 'Hamsadhvani', 'Jog', 'Kedar', 'Khamaj',
                 'Lalat', 'Madhukauns', 'Madhuvanti', 'Malkauns', 'Marubihag', 'Marwa', 'Miya malhar',
                 'Puriya dhanashree', 'Rageshri', 'Shree', 'Shuddh sarang', 'Todi', 'Yaman kalyan']

        labels = [ragas[idx] for idx in list(labels)]
        fig = px.scatter(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], color=labels, hover_data=[labels])
        fig.write_html("tsne.html")

        # plotter.save_figs('tsne.pdf')
        # parser = ArgumentParser()
        # parser.add_argument('--learning_rate', type=float, default=0.01)
        # parser.add_argument('--batch_size', type=int, default=self.batch_size)
        # hparams = parser.parse_args()
        #
        # self.phono_net = PhonoNet(network, None, None, hparams)
        # self.phono_net.load_state_dict(torch.load(
        #     '/home/raunak/cmu-ragaresearch/main_scripts/default/version_50/checkpoints/epoch=68.ckpt')[
        #                                    'state_dict'])
        #
        # dl = DataLoader(dataset, batch_size=100, num_workers=mp.cpu_count(),
        #                 pin_memory=True)
        #
        # embeddings = []
        # labels = []
        #
        # self.phono_net.cuda()
        # for x, y in tqdm(dl, desc="Evaluating Model"):
        #     embed = self.phono_net.generate_embedding(x.cuda()).detach().cpu()
        #     embeddings.append(embed)
        #     labels.append(y)
        # embeddings = torch.cat(embeddings, dim=0)
        # labels = torch.cat(labels, dim=0)
        # print(embeddings.shape)
        # print(labels.shape)
        # tsne_embeddings = TSNE().fit_transform(embeddings)
        # import pickle
        # pickle.dump((labels, embeddings), open('tsne.pkl', 'wb'))


if __name__ == '__main__':
    from torchsummary import summary

    pn = PhonoNet(None, None, hparams=None)
    summary(pn, (12, 15000))
