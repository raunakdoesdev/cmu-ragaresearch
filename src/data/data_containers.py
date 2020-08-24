from torch.utils.data import Dataset
import torch
import pickle
import glob
import os
import json
import copy
from tqdm.auto import tqdm

class FullChromaDataset(Dataset):
    """
    Dataset object that returns full length songs as chromagams, accompanied with a "RAGA ID" label which is
    computed dynamically based on the ragas in the given set in alphabetical order.
    """

    def _assign_raga_ids(self):
        """
        Helper function. Creates a raga to raga id mapping using the list of ragas in alphabetical order.
        """

        mbids = [os.path.basename(file_name).split('.')[0] for file_name in self.files]
        raga_ids = {self.metadata[mbid]['raags'][0]['common_name'] for mbid in mbids}
        raga_ids = sorted(raga_ids)
        self.raga_ids = {k: v for v, k in enumerate(raga_ids)}

    def _get_raga_id(self, file):
        """
        Helper function. Gets the raga id associated with a specific chromagram file.
        """

        if not hasattr(self, 'raga_ids') or self.raga_ids is None:
            self._assign_raga_ids()
        mbid = os.path.basename(file).split('.')[0]
        return self.raga_ids[self.metadata[mbid]['raags'][0]['common_name']]

    def __init__(self, json_path, data_folder, include_mbids=None):
        """
        Creates a new dataset object.

        :param json_path: path to json file with all raga metadata
        :param data_folder: folder with all of the chromagrams inside
        :param include_mbids: list of song ids to include
        """
        self.files = glob.glob(os.path.join(data_folder, '**/*.pkl'))
        self.files += glob.glob(os.path.join(data_folder, '*.pkl'))
        self.metadata = json.load(open(json_path, 'r'))

        # Remove files not on the "include" list (can easily create a subset of the main dataset)
        if include_mbids is not None:
            for self.file in copy.deepcopy(self.files):
                file_name = os.path.basename(self.file).split('.pkl')[0]
                if file_name not in include_mbids:
                    self.files.remove(self.file)
        else:
            for self.file in copy.deepcopy(self.files):
                mbid = os.path.basename(self.file).split('.')[0]
                if len(self.metadata[mbid]['raags']) < 1:
                    self.files.remove(self.file)

        self.X = []
        self.y = []
        for file in tqdm(self.files, desc="Loading Chromagram Files"):
            self.X.append(torch.FloatTensor(pickle.load(open(file, 'rb'))))
            self.y.append(self._get_raga_id(file))

    @classmethod
    def init_x_y(cls, X, y, raga_ids):
        """
        Helper method. Bypasses default constructor to allow for construction with just X and y objects directly.
        """
        obj = cls.__new__(cls)
        obj.X = X
        obj.y = y
        obj.raga_ids = raga_ids
        return obj

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)

    def train_test_split(self, test_size=None, train_size=None, random_state=1):
        """
        Creates two new datasets from the original dataset object by splitting the datasets in a stratified fashion.

        :param test_size: size of test set (as a percentage)
        :param train_size: size of the train set (as a percentage)
        :param random_state: random seed used to shuffle the data before splitting
        """

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, train_size=train_size,
                                                            stratify=self.y, random_state=random_state)
        return FullChromaDataset.init_x_y(X_train, y_train, self.raga_ids), FullChromaDataset.init_x_y(X_test, y_test,
                                                                                                       self.raga_ids)

    def greedy_split(self, train_size=None, test_size=None):
        if test_size is None:
            test_size = 1 - train_size
        else:
            train_size = 1 - test_size

        # Split Samples by Raga
        samples_by_raga = [[] for i in range(len(self.raga_ids))]
        for X, y in self:
            samples_by_raga[y].append(X)
        X_train, y_train = [], []
        X_test, y_test = [], []
        for raga, samples in enumerate(samples_by_raga):
            train_len, test_len = 0, 0
            for sample in sorted(samples, reverse=True, key=lambda sample: len(sample[0])):
                if train_len <= test_len:
                    X_train.append(sample)
                    y_train.append(raga)
                    train_len += len(sample[0]) * (test_size / train_size)
                else:
                    X_test.append(sample)
                    y_test.append(raga)
                    test_len += len(sample[0])

        return FullChromaDataset.init_x_y(X_train, y_train, self.raga_ids), FullChromaDataset.init_x_y(X_test, y_test,
                                                                                                       self.raga_ids)


class ChromaChunkDataset(Dataset):
    def __init__(self, full_chroma_dataset: FullChromaDataset, chunk_size, augmentation=None):
        """
        Class for chunkifying an existing full chroma dataset

        :param full_chroma_dataset: FullChromaDataset object
        :param chunk_size: size of the chunks to make from the original set
        :param augmentation: function that the chunks are passed through before calling get_item (user defined)
        """

        self.X = []
        self.y = []
        self.augmentation = augmentation
        self.raga_ids = full_chroma_dataset.raga_ids

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


if __name__ == '__main__':
    fcd = FullChromaDataset(json_path=config['data']['metadata'],
                            data_folder=config['data']['chroma_folder'],
                            include_mbids=json.load(open(config['data']['limit_songs'])))
    fcd_train, fcd_val = fcd.greedy_split(train_size=0.75)
    fcd_train_chunks = ChromaChunkDataset(fcd_train, chunk_size=config['data']['chunk_size'])
    assert (len(fcd_train_chunks) > len(fcd_train))
    assert 'Bhairav' in get_raga_list(fcd)
