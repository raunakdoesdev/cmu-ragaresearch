from easydict import EasyDict

from multiprocessing import Pool, cpu_count
import tqdm


def parallel_progress_run(fn, jobs, **kwargs):
    with Pool(cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(fn, jobs), total=len(jobs), **kwargs))
    return r


config = EasyDict()
