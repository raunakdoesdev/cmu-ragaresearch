from google.cloud import storage

# if not os.path.exists(config['data']['chroma_folder']):
#     raise Exception('Error Downloading Dataset!')
#     if not os.path.exists('chroma.zip'):
#         !gsutil -m cp gs://ragaresearch/chroma.zip ./
#     !unzip chroma.zip
#     !mv n_ftt4096__hop_length2048 hindustani-chroma
#     !rm chroma.zip