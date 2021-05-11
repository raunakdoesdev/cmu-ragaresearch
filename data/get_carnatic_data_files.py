from compmusic import dunya

dunya.set_token('ad57ef18f8c3a2f4962b7883ac6ed38b3578ba38')
a = dunya.carnatic.get_recordings(recording_detail=True)

import json

with open('carnatic.json', 'w') as fp:
    json.dump(a, fp)

def download_all_songs():
    if config.mode == 'carnatic':
        from compmusic.dunya.carnatic import get_recording

    with open(path, 'wb'):
        recording = get_recording(mbid)

    download_release(rel['mbid'], "./temp/")
