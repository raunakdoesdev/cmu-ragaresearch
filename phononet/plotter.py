import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['lines.linewidth'] = 2.5  # 2.5

# ragas = ['Abhogi', 'Ahir bhairav', 'Alahiya bilawal', 'Bageshree', 'Bairagi', 'Basant', 'Bhairav', 'Bhoop',
#          'Bihag', 'Bilaskhani todi', 'Darbari', 'Des', 'Gaud malhar', 'Hamsadhvani', 'Jog', 'Kedar', 'Khamaj',
#          'Lalat', 'Madhukauns', 'Madhuvanti', 'Malkauns', 'Marubihag', 'Marwa', 'Miya malhar',
#          'Puriya dhanashree', 'Rageshri', 'Shree', 'Shuddh sarang', 'Todi', 'Yaman kalyan']

ragas = ['Purvikalyani', 'ananda bhairavi', 'atana', 'begada', 'behag', 'bhairavi', 'bilahari', 'devagandhari',
         'dhanyasi', 'gaula', 'harikambhoji', 'huseni', 'kalyani', 'kamas', 'kamavardani', 'kamboji', 'kanada', 'kapi',
         'karaharapriya', 'kedaragaula', 'madyamavati', 'mayamalava gaula', 'mohanam', 'mukhari', 'nata', 'natakurinji',
         'riti gaula', 'sahana', 'sama', 'sankarabharanam', 'saveri', 'senchurutti', 'shanmukhapriya', 'shri',
         'shri ranjani', 'sindhubhairavi', 'surati', 'thodi', 'varali', 'yadukula kambhoji']


def plot_confusion_matrix(cm):
    fig, ax = make_fig()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    df_cm = pd.DataFrame(cm, index=ragas, columns=ragas)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, ax=ax, annot=True)
    ax.set_title("Confusion Matrix")
    # Matplotlib 3.11 Bug Fix:
    # (https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)


def plot_class_accuracies(class_accuracies):
    fig, ax = make_fig()
    sns.barplot(ragas, class_accuracies, ax=ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Raga Name')
    ax.set_xticklabels(ragas, rotation=90)


def plot_class_f1(f1_scores):
    fig, ax = make_fig()
    sns.barplot(ragas, f1_scores, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Raga Name')
    ax.set_xticklabels(ragas, rotation=90)


def save_figs(filename):
    fn = os.path.join(os.getcwd(), filename)
    pp = PdfPages(fn)
    for i in plt.get_fignums():
        plt.figure(i).tight_layout()
        pp.savefig(plt.figure(i))
        plt.close(plt.figure(i))
    pp.close()


def make_fig():
    fig, ax = plt.subplots()
    ax.grid(False)
    fig.tight_layout()
    return fig, ax
