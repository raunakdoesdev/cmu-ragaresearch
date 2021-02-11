import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from experiments.circular_padding.resnet_circular_padding import *
from models.custom_resnet import *
from src import *
from src.data.data_module import MusicDataModule

config = toml.load('carnatic.toml')
fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])),
                        carnatic=True)

train, fcd_not_train = fcd.greedy_split(train_size=0.70)
val, test = fcd_not_train.greedy_split(test_size=0.5)

train = ChromaChunkDataset(train, chunk_size=100, augmentation=transpose_chromagram, stride=10)
data = MusicDataModule(train, val, batch_size=32)

import sys

if sys.argv[1] == '34':
    model = ResNet34Circular(num_classes=max(fcd.y) + 1)
elif sys.argv[1] == '50':
    model = ResNet50Circular(num_classes=max(fcd.y) + 1)
elif sys.argv[1] == '101':
    model = ResNet101Circular(num_classes=max(fcd.y) + 1)

model.epochs = 50

logger = WandbLogger(project='Raga Benchmark', name=f'Carnatic Training - ResNet{sys.argv[1]}')

checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    filepath=f'/mnt/disks/checkpoints/new-checkpoints/carnatic-resnet{sys.argv[1]}-{{epoch:02d}}-{{val_accuracy:.2f}}',
    save_top_k=1,
    mode='max',
    verbose=True
)

if len(sys.argv) == 3:
    print('Resuming!')
    resume_from_checkpoint = os.path.join('/mnt/disks/checkpoints/new-checkpoints', sys.argv[2])
else:
    print('Not resuming!')
    resume_from_checkpoint = None

trainer = Trainer(gpus=1, logger=logger, max_epochs=model.epochs, num_sanity_val_steps=2,
                  deterministic=True, resume_from_checkpoint=resume_from_checkpoint,
                  val_check_interval=0.1, auto_lr_find=False, checkpoint_callback=checkpoint_callback)
model.lr = 0.1
trainer.fit(model, data)
