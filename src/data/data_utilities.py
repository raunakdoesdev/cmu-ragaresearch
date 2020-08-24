import random
import torch


def get_raga_list(dataset):
    """
    Accesses a FullChromaDataset or ChunkChromaDataset object and extracts list of ragas in order of raga id
    """
    return sorted(dataset.raga_ids)


def transpose_chromagram(x, shift=None):
    if shift is None:
        shift = random.randint(0, 11)
    if shift == 0:
        return x
    else:
        return torch.cat([x[-shift:, :], x[:-shift, :]], 0)


def chunk_chroma(chroma, chunk_size=1500):
    X = []
    unfolded = chroma.split(chunk_size, dim=1)
    for i in range(len(unfolded)):
        chroma = unfolded[i]
        if unfolded[i].shape[1] != chunk_size:
            padding = torch.zeros(unfolded[i].shape[0], chunk_size - unfolded[i].shape[1]).cuda()
            chroma = torch.cat((unfolded[i], padding), 1)
        X.append(chroma.unsqueeze(0))
    X = torch.cat(X, dim=0)
    return X
