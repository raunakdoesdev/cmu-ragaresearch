import compmusic
from compmusic import dunya

from src.util import config
import os


def download_mbid(mbid):
    dunya.set_token(config.dunya_token)
    save_path = os.path.join(config.chroma.audio_load_path, mbid + '.mp3')
    if not os.path.exists(save_path):
        contents = compmusic.dunya.docserver.get_mp3(mbid)
        with open(save_path, 'wb') as f:
            f.write(contents)
