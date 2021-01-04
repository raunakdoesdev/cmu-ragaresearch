import pytorch_lightning as pl
import toml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from models.custom_resnet import *
from src import *
from src.data.data_module import MusicDataModule

pl.seed_everything(42)

config = toml.load('config.toml')

fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

train, fcd_not_train = fcd.greedy_split(train_size=0.70)
val, test = fcd_not_train.greedy_split(test_size=0.5)

train = ChromaChunkDataset(train, chunk_size=100, augmentation=transpose_chromagram, stride=10)
data = MusicDataModule(train, val, batch_size=32)
model = ResNet34(num_classes=max(fcd.y) + 1)

logger = WandbLogger(project='ResnetPhononet')
trainer = Trainer(gpus=1, logger=logger, max_epochs=100000, num_sanity_val_steps=2, deterministic=True,
                  val_check_interval=0.1, auto_lr_find=False)
model.lr = 0.1
trainer.fit(model, data)
