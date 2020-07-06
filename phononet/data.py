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


def transpose_chromagram(x, shift=None):
    if shift is None:
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
    def init_x_y(cls, X, y):
        obj = cls.__new__(cls)
        obj.X = X
        obj.y = y
        return obj

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)

    def train_test_split(self, test_size=None, train_size=None, random_state=1):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, train_size=train_size,
                                                            stratify=self.y, random_state=random_state)
        return FullChromaDataset.init_x_y(X_train, y_train), FullChromaDataset.init_x_y(X_test, y_test)


class ChromaChunkDataset(Dataset):
    def __init__(self, full_chroma_dataset: FullChromaDataset, chunk_size, augmentation=None):
        """

        :param full_chroma_dataset:
        :param chunk_size:
        :param augmentation:
        """
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


import torch
import torch.utils.data
import torchvision


def dataset_get_label_callback(dataset, idx):
    return dataset[idx][1]


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=dataset_get_label_callback):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
