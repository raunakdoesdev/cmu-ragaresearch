import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os

from experiments.circular_padding.resnet_circular_padding import *
from models.phononet import Phononet
from src import *
from src.data.data_module import MusicDataModule

config = toml.load('hindustani.toml')
fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])),
                        compression=1)

train, fcd_not_train = fcd.greedy_split(train_size=0.70)
val, test = fcd_not_train.greedy_split(test_size=0.5)

train = ChromaChunkDataset(train, chunk_size=100, augmentation=transpose_chromagram, stride=10)
data = MusicDataModule(train, val, batch_size=32)

model = Phononet(num_classes=max(fcd.y) + 1)
model.epochs = 50

logger = WandbLogger(project='Raga Benchmark', name=f'Hindustani Training - PhonoNet')

checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    filepath=f'/mnt/disks/checkpoints/new-checkpoints/hindustani-phononet-{{epoch:02d}}-{{val_accuracy:.2f}}',
    save_top_k=1,
    mode='max',
    verbose=True
)

resume_from_checkpoint = None

trainer = Trainer(gpus=1, logger=logger, max_epochs=model.epochs, num_sanity_val_steps=2,
                  deterministic=True, resume_from_checkpoint=resume_from_checkpoint,
                  val_check_interval=0.1, auto_lr_find=False, checkpoint_callback=checkpoint_callback)

model.lr = 0.1
trainer.fit(model, data)
