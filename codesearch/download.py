# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from pathlib import Path
import urllib.request
import tarfile
import gzip
import zipfile
from io import BytesIO
from glob import glob

from tqdm import tqdm

from codesearch.data_config import MODELS, SNIPPET_COLLECTIONS, EVAL_DATASETS, TRAIN_DATASETS


def download_dataset(dataset):
    if dataset in SNIPPET_COLLECTIONS:
        dataset = SNIPPET_COLLECTIONS[dataset]
    elif dataset in EVAL_DATASETS:
        dataset = EVAL_DATASETS[dataset]
    elif dataset in TRAIN_DATASETS:
        dataset = TRAIN_DATASETS[dataset]
    else:
        raise ValueError(f"dataset must be one of {' '.join(list(SNIPPET_COLLECTIONS) + list(EVAL_DATASETS))}")
    path = dataset.get("path", None) or dataset["path_pattern"]
    if glob(str(path)):
        print(f"Dataset path already exists. The dataset is probably already downloaded.")
        return 
    print(f"Downloading dataset from {dataset['url']}")
    
    # Sometimes a zip/tar archive contains multiple datasets
    target_path = dataset.get("extract_at", None) or path
    download_and_extract(dataset["url"], Path(target_path))
    
        
def download_model(modelname):
    model = MODELS.get(modelname, None)
    if not model:
        raise ValueError(f"model must be one of {' '.join(list(MODELS))}")
    
    target_dir = Path(model["path"])
    if (target_dir/"class.txt").exists():
        print(f"The model path already exists. The model is probably already downloaded")
        return
    if not target_dir.exists():
        target_dir.mkdir()
    print(f"Downloading {modelname} model from {model['url']}")
    download_and_extract_tar(model["url"], target_dir)
    
def download_and_extract_tar(url, target_dir):
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir()
    tempfile = url2tempfile(url)
    with gzip.GzipFile(tempfile) as funzip:
        with tarfile.TarFile(fileobj=funzip) as funtar:
            funtar.extractall(target_dir)
    
    
def download_and_extract_gz(url, filename):
    parentdir = Path(filename).parent
    if not parentdir.exists():
        parentdir.mkdir()
    tempfile = url2tempfile(url)
    with gzip.GzipFile(tempfile) as funzip:
        with open(filename, "wb") as f:
            f.write(funzip.read())
        

def download_and_extract_zip(url, filename):
    tempfile = url2tempfile(url)
    funzip = zipfile.ZipFile(tempfile, "r")
    funzip.extractall(filename)
    funzip.close()
            

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
        
    
def url2tempfile(url):
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:  
        tempfile, _ = urllib.request.urlretrieve(url, reporthook=t.update_to, data=None)
        t.total = t.n
    return tempfile
        
            
def download_and_extract(url, filename):
    urlbase = url[:-len("?download=1")] if url.endswith("?download=1") else url
    if urlbase.endswith(".gz"):
        download_and_extract_gz(url, filename)
    elif urlbase.endswith(".zip"):
        download_and_extract_zip(url, filename)
    else:
        raise ValueError(f"Extension of {url} not supported")
    