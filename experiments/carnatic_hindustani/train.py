from multiprocessing import cpu_count

import pytorch_lightning as pl
import toml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset

from models.blocked_phononet import BlockedPhononet
from src import *

pl.seed_everything(42)

config = toml.load('carnatic.toml')
fcd_carnatic = FullChromaDataset(json_path=config['data']['metadata'],
                                 data_folder=config['data']['chroma_folder'],
                                 include_mbids=json.load(open(config['data']['limit_songs'])), carnatic=True)
train_carnatic, fcd_not_train_carnatic = fcd_carnatic.train_test_split(train_size=0.70)
val_carnatic, test_carnatic = fcd_not_train_carnatic.train_test_split(test_size=0.5)

chunk_sizes = (100,)
strides = (10, 10, 10)
chunked_data = [ChromaChunkDataset(train_carnatic, chunk_size=chunk_size, stride=stride)
                for chunk_size, stride in zip(chunk_sizes, strides)]
train_carnatic = ConcatDataset(chunked_data)

config = toml.load('hindustani.toml')
fcd_hindustani = FullChromaDataset(json_path=config['data']['metadata'],
                                   data_folder=config['data']['chroma_folder'],
                                   include_mbids=json.load(open(config['data']['limit_songs'])),
                                   raga_id_offset=max(fcd_carnatic.y) + 1)  # shift all of the classes up

train_hindustani, fcd_not_train_hindustani = fcd_hindustani.train_test_split(train_size=0.70)
val_hindustani, test_hindustani = fcd_not_train_hindustani.train_test_split(test_size=0.5)

chunked_data = [ChromaChunkDataset(train_hindustani, chunk_size=chunk_size, stride=stride)
                for chunk_size, stride in zip(chunk_sizes, strides)]
train_hindustani = ConcatDataset(chunked_data)

train = ConcatDataset([train_hindustani, train_carnatic])
val = ConcatDataset([val_hindustani, val_carnatic])

chunk_sizes = (50, 75, 100)


def blocked_collate(batch):
    separated = {chunk_size: [] for chunk_size in chunk_sizes}
    for x, y in batch:
        separated[x.shape[1]].append((x, torch.LongTensor([y])))
    ret = []
    for chunk_size in chunk_sizes:
        if len(separated[chunk_size]) > 0:
            x = torch.stack([x for x, y in separated[chunk_size]])
            y = torch.stack([y for x, y in separated[chunk_size]]).squeeze(1)  # remove singleton dim
            ret.append((x, y))
        else:
            ret.append([])
    return ret


class BlockedMusicDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        return DataLoader(train, batch_size=32, shuffle=True,
                          num_workers=cpu_count(),
                          collate_fn=blocked_collate,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(val, batch_size=1, num_workers=cpu_count(),
                          pin_memory=True)


print(max(fcd_hindustani.y) + 1)
data = BlockedMusicDataModule()
model = BlockedPhononet(num_out=max(fcd_hindustani.y) + 1)

logger = WandbLogger(project='CarnaticPhononet', name='Hindustani + Carnatic')
trainer = Trainer(logger=logger, max_epochs=100000, num_sanity_val_steps=2, deterministic=True,
                  val_check_interval=0.1, auto_lr_find=False)
model.lr = 0.000912
trainer.fit(model, data)
