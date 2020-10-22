import toml
import pytorch_lightning as pl
import pytorch_lightning.metrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.metrics.functional import *
import plotly.express as px
import numpy as np
from models.blocked_phononet import BlockedPhononet
from torch.utils.data import DataLoader
from src import *
from src.data.data_module import MusicDataModule

config = toml.load('config.toml')

fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

train, fcd_not_train = fcd.train_test_split(train_size=0.70)
val, test = fcd_not_train.train_test_split(test_size=0.5)

data = MusicDataModule(train, val, test_set=test, batch_size=1)

model = BlockedPhononet()
logger = WandbLogger(project='Chunking', name=f'Simple Blocked PhonoNet')
trainer = Trainer(gpus=1, logger=logger, max_epochs=100000, num_sanity_val_steps=0, auto_lr_find='lr')
# model.hparams.lr = 0.03
# print(f'Setting LR to {model.hparams.lr}')
trainer.fit(model, data)
