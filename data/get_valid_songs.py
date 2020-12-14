import requests
import json

r = requests.get('https://raw.githubusercontent.com/sankalpg/MelodicAnalysisDatasets/master/RagaDataset/Carnatic/_info_/path_mbid_ragaid.json')
r = r.json()

with open('carnatic-limited-songs-list.json', 'w') as fp:
    json.dump(list(r.keys()), fp)