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

chunk_size = 750

fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

fcd_train, fcd_not_train = fcd.train_test_split(train_size=0.70)
fcd_val, fcd_test = fcd_not_train.train_test_split(test_size=0.5)

train = ChromaChunkDataset(fcd_train, chunk_size=chunk_size, augmentation=transpose_chromagram, stride=chunk_size)
val = ChromaChunkDataset(fcd_val, chunk_size=chunk_size, chunkify=False)
data = MusicDataModule(train, val, test_set=fcd_val, batch_size=config['training']['batch_size'])

model = BlockedPhononet()
model.hparams.chunk_size = chunk_size
logger = WandbLogger(project='Chunking', name=f'CS=500,No Overlap')
trainer = Trainer(gpus=1, logger=logger, max_epochs=100000)
model.hparams.lr = 0.03
print(f'Setting LR to {model.hparams.lr}')
trainer.fit(model, data)

