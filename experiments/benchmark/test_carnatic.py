import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.custom_resnet import *
from src import *
from src.data.data_module import MusicDataModule

config = toml.load('carnatic.toml')
fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])), carnatic=True)

print(len(fcd))
train, fcd_not_train = fcd.greedy_split(train_size=0.70)
val, test = fcd_not_train.greedy_split(test_size=0.5)

train = ChromaChunkDataset(train, chunk_size=100, augmentation=transpose_chromagram, stride=10)
data = MusicDataModule(train, val,  test, batch_size=32)
model = ResNet101(num_classes=max(fcd.y) + 1)


def save_for_processing(y_score, y_pred, y_true):
    with open('saves/carnatic-test-out.pkl', 'wb') as f:
        torch.save([y_score, y_pred, y_true], f)


model.test_visualizations = save_for_processing

trainer = Trainer(gpus=1, deterministic=True,
                  resume_from_checkpoint='checkpoints/carnatic101-training-epoch=46-val_accuracy=0.74.ckpt')
trainer.test(model=model, datamodule=data)
