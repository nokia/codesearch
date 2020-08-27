# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import os
from pydoc import locate
from pathlib import Path
from codesearch.data_config import MODELS
from codesearch.download import download_model

import dill
from tqdm import tqdm

class Saveable(object):
    
    def save(self, directory):
        directory = Path(directory)
        if not directory.exists():
            os.makedirs(directory)
        with open(directory/"class.txt", "w") as f:
            f.write(".".join((self.__module__, self.__class__.__name__)))
    
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory/"class.txt", "r") as f:
            classname = next(f).strip()
            print(classname)
            cls_ = locate(classname)
        return cls_.load(directory)
    

class SaveableFunction(Saveable):
    
    def __init__(self, fn):
        self.fn = fn
        
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
        
    def save(self, path):
        super().save(path)
        path = Path(path)
        with open(path/"saveable_func.pkl", "wb") as f:
            dill.dump(self, f)
        
    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path/"saveable_func.pkl", "rb") as f:
            return dill.load(f)
        
        
def load_model(model_name):
    if Path(model_name).exists():
        return Saveable.load(model_name)
    if not model_name in MODELS:
        raise ValueError(f"model_name should be one of {' '.join(list(MODELS.keys()))} or a path to a model on disk.")
    path = MODELS[model_name]["path"]
    if not path.exists():
         download_model(model_name)
    return Saveable.load(path)
    
def get_best_device():
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def progress_generate(l):
    return tqdm(l)