import pandas as pd
import toml

from src import *

config = toml.load('carnatic.toml')
fcd_carnatic = FullChromaDataset(json_path=config['data']['metadata'],
                                 data_folder=config['data']['chroma_folder'],
                                 include_mbids=json.load(open(config['data']['limit_songs'])), carnatic=True)
train_carnatic, fcd_not_train_carnatic = fcd_carnatic.greedy_split(train_size=0.70)
val_carnatic, test_carnatic = fcd_not_train_carnatic.greedy_split(test_size=0.5)

config = toml.load('hindustani.toml')
fcd_hindustani = FullChromaDataset(json_path=config['data']['metadata'],
                                   data_folder=config['data']['chroma_folder'],
                                   include_mbids=json.load(open(config['data']['limit_songs'])))

train_hindustani, fcd_not_train_hindustani = fcd_hindustani.greedy_split(train_size=0.70)
val_hindustani, test_hindustani = fcd_not_train_hindustani.greedy_split(test_size=0.5)

gen_list = [
    ('train_carnatic', train_carnatic),
    ('val_carnatic', val_carnatic),
    ('test_carnatic', test_carnatic),
    ('train_hindustani', train_hindustani),
    ('val_hindustani', val_hindustani),
    ('test_hindustani', test_hindustani)
]

for name, dset in gen_list:
    raga_ids = {v: k for k, v in dset.raga_ids.items()}  # raga id to raga name mapping
    train_df = pd.DataFrame({'MBID': dset.mbids,
                             'Raga': [raga_ids[y].title() for y in dset.y]},
                            columns=['MBID', 'Raga'])
    train_df.to_csv(f'{name}.csv', index=False)
