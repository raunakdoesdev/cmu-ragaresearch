import toml

from data.pipeline.download_songs import download_mbid
from data.pipeline.generate_chroma import get_mp3_list, generate_chroma, get_mbid_list
from src.util import config
from tqdm.auto import tqdm
from p_tqdm import p_map
import warnings
import os

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config.update(toml.load('carnatic3.toml'))
    p_map(generate_chroma, get_mp3_list())  # convert to pickle files
    os.system('sudo shutdown now')
