import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from src import *
from src.data.data_module import MusicDataModule

circular_padding = True




config = toml.load('hindustani.toml')
fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

train, fcd_not_train = fcd.greedy_split(train_size=0.70)
val, test = fcd_not_train.greedy_split(test_size=0.5)

train = ChromaChunkDataset(train, chunk_size=100, augmentation=transpose_chromagram, stride=10)
data = MusicDataModule(train, val, test, batch_size=32)
model = ResNet50(num_classes=max(fcd.y) + 1)

if circular_padding:
    model = ResNet50(num_classes=max(fcd.y) + 1)

    pass
else:
    from models.custom_resnet import ResNet50

def save_for_processing(y_score, y_pred, y_true):
    with open('saves/hindustani-test-out.pkl', 'wb') as f:
        torch.save([y_score, y_pred, y_true], f)


model.test_visualizations = save_for_processing

trainer = Trainer(gpus=1, deterministic=True,
                  resume_from_checkpoint='checkpoints/hindustani-training-epoch=29-val_accuracy=0.90.ckpt')
trainer.test(model=model, datamodule=data)


# economics, psychology, statistics, applied math, computer science