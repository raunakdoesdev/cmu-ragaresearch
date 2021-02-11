import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os

from experiments.circular_padding.resnet_circular_padding import ResNet34Circular
from models.custom_resnet import *
from src import *
from src.data.data_module import MusicDataModule

config = toml.load('hindustani.toml')
fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

train, fcd_not_train = fcd.greedy_split(train_size=0.70)
val, test = fcd_not_train.greedy_split(test_size=0.5)

train = ChromaChunkDataset(train, chunk_size=100, augmentation=None, stride=10)
data = MusicDataModule(train, val, batch_size=32)

mode = 'circular'

if mode == 'circular':
    model = ResNet34Circular(num_classes=max(fcd.y) + 1)
else:
    model = ResNet34(num_classes=max(fcd.y) + 1)

import torchsummary

torchsummary.summary(model, (1, 12, 2400), device='cpu')

model.epochs = 50

logger = WandbLogger(project='Raga Benchmark', name='Circular (Fixed)')

trainer = Trainer(gpus=1, logger=logger, max_epochs=model.epochs, num_sanity_val_steps=2, deterministic=True,
                  val_check_interval=0.1, auto_lr_find=False)
model.lr = 0.1
trainer.fit(model, data)
