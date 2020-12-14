import json

old_metadata = json.load(open('carnatic.json', 'r'))

# Reorganize into dictionary format to make lookups blazing fast!
new_metadata = {}
for item in old_metadata:
    new_metadata[item['mbid']] = item

with open('carnatic-metadata.json', 'w') as fp:
    json.dump(new_metadata, fp, indent=2)
