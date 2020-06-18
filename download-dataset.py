from tqdm import tqdm
from compmusic import dunya
from compmusic.dunya import hindustani, get_mp3
import json
import multiprocessing as mp
import sys
import requests
import numpy as np
import os


def download(url, fname, position=0):
    try:
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
                desc=fname,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                position=position,
                leave=False
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception:
        with open('errors.txt', 'a') as f:
            f.write(f'Failed to download {url}\n')


def download_recording(args):
    idx, chunk = args
    for recording in chunk:
        if not os.path.exists(f'hindustani/audiodata/{recording["mbid"]}.mp3'):
            download(f'https://dunya.compmusic.upf.edu/document/by-id/{recording["mbid"]}/mp3',
                     f'hindustani/audiodata/{recording["mbid"]}.mp3', position=idx + 1)


if __name__ == '__main__':
    dunya.set_token('ad57ef18f8c3a2f4962b7883ac6ed38b3578ba38')
    hindustani_metadata = json.load(open('hindustani.json', 'r'))

    num_procs = mp.cpu_count() * 3
    chunks = np.array_split(hindustani_metadata, num_procs)

    with mp.Pool(mp.cpu_count(), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        r = list(tqdm(p.imap(download_recording, zip(range(num_procs), chunks)),
                      total=len(hindustani_metadata), desc="Master Download Progress", position=0))
