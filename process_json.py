import json
from tqdm import tqdm

hindustani_metadata = json.load(open('hindustani.json', 'r'))
new_hindustani_metadata = {}

for recording in tqdm(hindustani_metadata):
    new_hindustani_metadata[recording['mbid']] = recording

json.dump(new_hindustani_metadata, open('hindustaniv2.json', 'w'))
