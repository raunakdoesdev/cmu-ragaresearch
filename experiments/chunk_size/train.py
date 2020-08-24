import toml
import pytorch_lightning as pl
import pytorch_lightning.metrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.metrics.functional import *
import plotly.express as px
import numpy as np
from models.phononet import Phononet
from src import *
from src.data.data_module import MusicDataModule

config = toml.load('config.toml')

for chunk_size in [375]:
    fcd = FullChromaDataset(json_path=config['data']['metadata'],
                            data_folder=config['data']['chroma_folder'],
                            include_mbids=json.load(open(config['data']['limit_songs'])))

    fcd_train, fcd_not_train = fcd.greedy_split(train_size=0.70)
    fcd_val, fcd_test = fcd_not_train.greedy_split(test_size=0.5)

    train = ChromaChunkDataset(fcd_train, chunk_size=chunk_size, augmentation=transpose_chromagram)
    val = ChromaChunkDataset(fcd_val, chunk_size=chunk_size)
    data = MusicDataModule(train, val, test_set=fcd_val, batch_size=config['training']['batch_size'])

    model = Phononet()
    model.hparams.chunk_size = chunk_size
    logger = WandbLogger(project='Stage2 Chunk Size Tuning', name=f'{chunk_size}')
    trainer = Trainer(gpus=1, logger=logger, max_epochs=75,
                      checkpoint_callback=ModelCheckpoint(monitor='val_accuracy', mode='max'), auto_lr_find=False)


    def test_visualizations(y_score, y_pred, y_true):
        cm = confusion_matrix(y_pred, y_true).cpu().numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = px.bar(x=get_raga_list(fcd), y=100 * cm.diagonal(), labels={'x': 'Raga Name', 'y': 'Accuracy/Recall'})
        wandb.log({"class accuracy": fig})

        f1 = pl.metrics.sklearns.F1(average=None)
        fig = px.bar(x=get_raga_list(fcd), y=f1(y_true.cpu(), y_pred.cpu()).numpy(),
                     labels={'x': 'Raga Name', 'y': 'F1 Score'})
        fig.update_layout(yaxis=dict(range=[0, 1]))
        wandb.log({"class f1": fig})

        # Plot Confusion Matrix
        cm = confusion_matrix(y_pred, y_true).cpu().numpy()
        fig = px.imshow(cm, x=get_raga_list(fcd), y=get_raga_list(fcd),
                        labels=dict(x="Predicted Raga", y="Actual Raga", color="# of Chunks"), )
        fig.update_layout(
            autosize=False, width=1000, height=1000)
        wandb.log({'confusion matrix': fig})
        wandb.log({'Stage2 Accuracy': accuracy(y_pred, y_true)})
        wandb.log({'Stage2 F1': f1_score(y_pred, y_true)})


    trainer.fit(model, data)
    model.test_visualizations = test_visualizations
    trainer.test(model, datamodule=data)
