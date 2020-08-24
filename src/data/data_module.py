import pytorch_lightning as pl
from torch.utils.data import DataLoader


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set=None, batch_size=32):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = 32

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4,
                          pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader() if self.test_set is None else DataLoader(self.test_set, batch_size=1)
