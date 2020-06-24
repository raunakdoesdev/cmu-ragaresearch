import copy
import glob
import json
import os
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import Dataset


def transpose_chromagram(x):
    shift = random.randint(0, 11)
    if shift == 0:
        return x
    else:
        return torch.cat([x[-shift:, :], x[:-shift, :]], 0)


class FullChromaDataset(Dataset):
    def _assign_raga_ids(self):
        mbids = [os.path.basename(file_name).split('.')[0] for file_name in self.files]
        raga_ids = {self.metadata[mbid]['raags'][0]['common_name'] for mbid in mbids}
        raga_ids = sorted(raga_ids)
        self.raga_ids = {k: v for v, k in enumerate(raga_ids)}

    def _get_raga_id(self, file):
        if not hasattr(self, 'raga_ids') or self.raga_ids is None:
            self._assign_raga_ids()
        mbid = os.path.basename(file).split('.')[0]
        return self.raga_ids[self.metadata[mbid]['raags'][0]['common_name']]

    def __init__(self, json_path, data_folder, include_mbids=None):
        self.files = glob.glob(os.path.join(data_folder, '**/*.pkl'))
        self.files += glob.glob(os.path.join(data_folder, '*.pkl'))
        self.metadata = json.load(open(json_path, 'r'))

        # Remove files not on the "include" list (can easily create a subset of the main dataset)
        if include_mbids is not None:
            for self.file in copy.deepcopy(self.files):
                file_name = os.path.basename(self.file).split('.pkl')[0]
                if file_name not in include_mbids:
                    self.files.remove(self.file)

        self.X = []
        self.y = []
        for file in tqdm(self.files, desc="Loading Chromagram Files"):
            self.X.append(torch.FloatTensor(pickle.load(open(file, 'rb'))))
            self.y.append(self._get_raga_id(file))

    @classmethod
    def init_x_y(cls, X, y):
        obj = cls.__new__(cls)
        obj.X = X
        obj.y = y
        return obj

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)

    def train_test_split(self, test_size=None, train_size=None):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, train_size=train_size,
                                                            stratify=self.y)
        return FullChromaDataset.init_x_y(X_train, y_train), FullChromaDataset.init_x_y(X_test, y_test)


class ChromaChunkDataset(Dataset):
    def __init__(self, full_chroma_dataset: FullChromaDataset, chunk_size, augmentation=None):
        self.X = []
        self.y = []
        self.augmentation = augmentation
        for chroma, raga_id in full_chroma_dataset:
            unfolded = chroma.split(chunk_size, dim=1)
            for i in range(len(unfolded)):
                chroma = unfolded[i]
                if unfolded[i].shape[1] != chunk_size:
                    padding = torch.zeros(unfolded[i].shape[0], chunk_size - unfolded[i].shape[1])
                    chroma = torch.cat((unfolded[i], padding), 1)
                self.X.append(chroma.unsqueeze(0))
            self.y += len(unfolded) * [raga_id]

        self.X = torch.cat(self.X, dim=0)

    def __getitem__(self, item):
        if self.augmentation is None:
            return self.X[item], self.y[item]
        else:
            return self.augmentation(self.X[item]), self.y[item]

    def __len__(self):
        return len(self.y)
