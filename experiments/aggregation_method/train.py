import argparse
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
from torch.distributions import Categorical

config = toml.load('config.toml')

fcd = FullChromaDataset(json_path=config['data']['metadata'],
                        data_folder=config['data']['chroma_folder'],
                        include_mbids=json.load(open(config['data']['limit_songs'])))

fcd_train, fcd_not_train = fcd.greedy_split(train_size=0.70)
fcd_val, fcd_test = fcd_not_train.greedy_split(test_size=0.5)

train = ChromaChunkDataset(fcd_train, chunk_size=config['training']['chunk_size'],
                           augmentation=transpose_chromagram)
val = ChromaChunkDataset(fcd_val, chunk_size=config['training']['chunk_size'])

plot_num = 0

dfs = []


def log_pred_to_plot(x, y_true):
    global plot_num
    global dfs

    ragas = get_raga_list(fcd)
    x = torch.exp(x)
    rows = []
    for i, pred_x in enumerate(x):
        for raga_id in range(len(ragas)):
            rows.append([i, ragas[raga_id], float(pred_x[raga_id].cpu()), plot_num, ragas[y_true]])

    import pandas as pd
    df = pd.DataFrame(rows, columns=('Chunk', 'Raga', 'Probability', 'Test Song #', 'Correct Song'))
    dfs.append(df)
    fig = px.bar(df, x='Raga', y='Probability', color='Chunk', title=ragas[y_true])
    wandb.log({f'{ragas[y_true]} Test Song': fig})
    plot_num += 1


def shannon_entropy_agg(x):
    total_entropy = 0
    sum_result = torch.zeros(x[0].shape).cuda()
    for pred_x in x:
        entropy = 1 / torch.sum(pred_x * torch.exp(pred_x))
        total_entropy += entropy
        sum_result += pred_x * entropy
    final = (sum_result / total_entropy).unsqueeze(0)
    return final


def shannon_entropy_agg_raw_prob(x):
    total_entropy = 0
    sum_result = torch.zeros(x[0].shape).cuda()
    for pred_x in x:
        entropy = 1 / torch.sum(pred_x * torch.exp(pred_x))
        total_entropy += entropy
        sum_result += torch.exp(pred_x) * entropy
    final = (sum_result / total_entropy).unsqueeze(0)
    return final


def mean_agg(x, y_true):
    agg = torch.mean(x, dim=0, keepdim=True)
    _, y_pred = torch.max(agg, 1)
    if y_pred != y_true:
        log_pred_to_plot(x, y_true)
    return agg


def ranked_choice_agg(x, y_true):
    import torch.nn.functional as F
    ignores = []
    x = x[:, :30]  # get rid of extra outputs
    for i in range(25):
        print(x)
        agg = torch.mean(x, dim=0, keepdim=True)
        for ignore in ignores:
            agg[:, ignore] = float("inf")

        _, y_pred = torch.min(agg, 1)
        print(f"It's definitely not Raga #{y_pred}")

        ignores.append(y_pred)
        for ignore in ignores:
            x[:, ignore] = -float("inf")
        x = F.log_softmax(x, dim=1)

    agg = torch.mean(x, dim=0, keepdim=True)
    for ignore in ignores:
        agg[:, ignore] = -float("inf")
    return agg


def mean_agg_raw_prob(x):
    return torch.mean(torch.exp(x), dim=0, keepdim=True)


for agg_fn in [ranked_choice_agg]:
    data = MusicDataModule(train, val, test_set=fcd_val, batch_size=config['training']['batch_size'])

    model = Phononet.load_from_checkpoint(
        '/home/jupyter/refactor/experiments/aggregation_method/Stage2 Aggregation Tuning/1haeedjg/checkpoints/epoch=27.ckpt')
    if isinstance(model.hparams, dict):
        hparams = argparse.Namespace(**model.hparams)

    model.hparams.chunk_size = config['training']['chunk_size']

    model.hparams.agg_fn = agg_fn.__name__
    model.aggregation_fn = agg_fn

    logger = WandbLogger(project='Stage2 Aggregation Tuning', name="Visualize Predictions")
    trainer = Trainer(gpus=1, logger=logger, max_epochs=75,
                      checkpoint_callback=ModelCheckpoint(monitor='val_accuracy', mode='max'), auto_lr_find=False)


    # trainer.fit(model, data)
    def top_n_accuracy(preds, truths, n):
        best_n = np.argsort(preds, axis=1)[:, -n:]
        ts = np.argmax(truths, axis=1)
        successes = 0
        for i in range(ts.shape[0]):
            if ts[i] in best_n[i, :]:
                successes += 1
        return float(successes) / ts.shape[0]


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

        y_score = y_score.cpu()
        y_true = y_true.cpu()
        total = 0
        true = 0
        for i in range(len(y_true)):
            probs = y_score[i]
            actual = y_true[i]
            best_n = np.argsort(probs)[-5:]
            true += 1 if actual in best_n else 0
            total += 1

        wandb.log({'Stage 2 Top-3 Accuracy': true / total})


    model.test_visualizations = test_visualizations
    trainer.test(model, datamodule=data)

    # import pandas as pd
    # wandb.log({'my_data': wandb.Table(dataframe=pd.concat(dfs))})
