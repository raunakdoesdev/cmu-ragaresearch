from src.util import config, parallel_progress_run
import librosa
import os
import pickle
import json


def generate_chroma(mp3_file):
    save_path = os.path.join(config.chroma.chroma_save_path, os.path.basename(mp3_file).replace('.mp3', '.pkl'))

    if not os.path.exists(save_path):
        try:
            y, sr = librosa.load(mp3_file)
        except:
            return 'Failed'
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        with open(save_path, 'wb') as f:
            pickle.dump(chroma, f)
    print('Done')
    return 'Done'


def get_mbid_list():
    with open(config.chroma.limited_songs_list, 'r') as f:
        song_list = json.load(f)
    return song_list


def get_mp3_list():
    mbid_list = get_mbid_list()

    files = os.listdir(config.chroma.audio_load_path)
    return sorted([os.path.join(config.chroma.audio_load_path, file)
                   for file in files if file.endswith('.mp3') and file.replace('.mp3', '') in mbid_list])
