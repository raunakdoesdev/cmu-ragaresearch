import glob
import logging
import os
from pathlib import Path
import gdown


def extract_zip(zip_path):
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1e10:
        raise Exception('You need to download the Raw Dataset before Extraction!')
    target_dir = zip_path.replace('.zip', '')
    if not os.path.exists(target_dir):
        os.system(f'unzip {zip_path} -d {target_dir}')
    else:
        logging.info(f'{target_dir} already exist. Not repeating extraction.')
    return target_dir


def gdrive_download(url, out):
    if not os.path.exists(out) or os.path.getsize(out) < 1e10:
        return os.path.abspath(gdown.download(url, out))
    logging.info(f'{out} has already been downloaded. Not repeating download.')
    return os.path.abspath(out)


def recursive_grab(folder, skip_words=(), required_words=(), ext='mp3'):
    """
    Recursively grab files from a folder.
    :param folder:
    :param skip_words:
    :param required_words:
    :param ext:
    :return:
    """
    if folder is None or not os.path.exists(folder):
        raise IOError('Invalid video folder input!')
    file_paths = [str(file_path) for file_path in Path(folder).glob(f'**/*.{ext}')]

    for file_path in file_paths:
        for skip_word in skip_words:
            if skip_word in file_path:
                file_paths.remove(file_path)
                break
        for required_word in required_words:
            if required_word not in file_path:
                file_paths.remove(file_path)

    return file_paths
