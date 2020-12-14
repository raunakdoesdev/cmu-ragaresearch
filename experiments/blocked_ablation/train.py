from multiprocessing import cpu_count

import pytorch_lightning as pl
import toml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset

from models.blocked_phononet import BlockedPhononet
from src import *

pl.seed_everything(42)

config = toml.load('config.toml')

fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

train, fcd_not_train = fcd.train_test_split(train_size=0.70)
val, test = fcd_not_train.train_test_split(test_size=0.5)

chunk_sizes = (50, 75, 100,)
strides = (10, 10, 10,)
chunked_data = [ChromaChunkDataset(train, chunk_size=chunk_size, stride=stride)
                for chunk_size, stride in zip(chunk_sizes, strides)]

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


train = ConcatDataset(chunked_data)


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



data = BlockedMusicDataModule()
model = BlockedPhononet()

logger = WandbLogger(project='BlockedPhononetAblation', name='Full + No Dropout')
trainer = Trainer(gpus=1, logger=logger, max_epochs=100000, num_sanity_val_steps=2, deterministic=True,
                  val_check_interval=0.1, auto_lr_find=False)
model.lr = 0.000912
trainer.fit(model, data)

