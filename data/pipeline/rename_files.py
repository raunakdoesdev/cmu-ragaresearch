from src.util import config
import json
import toml
import os


def move_files():
    with open(config.convert.json_path, 'r') as f:
        songs = json.load(f)

    for song_dict in songs.values():
        dest_path = os.path.join(config.chroma.audio_load_path, song_dict['mbid'] + '.mp3')
        if not os.path.exists(dest_path):
            base_path = '/'.join(song_dict['path'].split('/')[3:]) + '.mp3'
            mp3_path = os.path.join(config.convert.gdrive_audio_path, base_path)
            os.system(f'ln -s {mp3_path} {dest_path}')


config.update(toml.load('carnatic3.toml'))
move_files()
