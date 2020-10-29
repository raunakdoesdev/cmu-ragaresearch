import toml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from models.phononet import Phononet
from src import *
from src.data.data_module import MusicDataModule

config = toml.load('config.toml')

fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

fcd_train, fcd_not_train = fcd.train_test_split(train_size=0.70)
fcd_val, fcd_test = fcd_not_train.train_test_split(test_size=0.5)

train = ChromaChunkDataset(fcd_train, chunk_size=75, stride=75)
val = ChromaChunkDataset(fcd_val, chunk_size=75, stride=75)

data = MusicDataModule(fcd_train, fcd_val, test_set=fcd_test, batch_size=1)

model = Phononet()
logger = WandbLogger(project='Chunking', name=f'Accumualte Gradients')
trainer = Trainer(gpus=1, logger=logger, max_epochs=100000, num_sanity_val_steps=0, #auto_lr_find='lr',
                  accumulate_grad_batches=100)
# model.hparams.lr = 0.03
# print(f'Setting LR to {model.hparams.lr}')
trainer.fit(model, data)
