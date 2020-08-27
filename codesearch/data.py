# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import json
from collections import defaultdict
from pathlib import Path
import gzip
import shutil
import re
import tarfile
import pickle
from glob import glob

from codesearch.data_config import MODELS, SNIPPET_COLLECTIONS, EVAL_DATASETS, TRAIN_DATASETS, CODE_FIELD, DESCRIPTION_FIELD
from codesearch.download import download_dataset


def load_jsonl(fn):
    snippets = []
    if fn.suffix == ".gz":
        open_ = gzip.open
    else:
        open_ = open
    with open_(fn) as f:
        for l in f:    
            snippets.append(json.loads(l))
    return snippets

def save_jsonl(fn, json_list):
    with open(fn, 'w') as f:
        for e in json_list:
            f.write(json.dumps(e))
            f.write('\n')

def load_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)

def _load_data(dataset_name, datasets):
    if not dataset_name in datasets:
        raise ValueError(f"collection_name should be one of {' '.join(list(datasets.keys()))}")
    ds = datasets[dataset_name]
    if "path" in ds:
        paths = [Path(ds["path"])]
    else:
        paths = [Path(p) for p in glob(str(ds["path_pattern"]))]
    if not paths or not paths[0].exists():
        download_dataset(dataset_name)
    
    data = []
    for path in paths:
        if str(path).endswith(".jsonl") or  str(path).endswith(".jsonl.gz"):
            data.extend(load_jsonl(path))   
        elif str(path).endswith(".pkl"):
            obj = load_pickle(path)
            if isinstance(obj, list):
                 data.extend(obj)
            else:
                 data.append(obj)
        else:
            data.append(str(path))
    return data

def load_snippet_collection(collection_name):
    
    snippets = _load_data(collection_name, SNIPPET_COLLECTIONS)
    if "codesearchnet" in collection_name:
        lang = collection_name.split("-")[1]
        for s in snippets:
            d = s.get("docstring_summary", s["docstring"])
            s[DESCRIPTION_FIELD] = d.split("\n")[0]
            c = s.get("function", s["code"])
            s[CODE_FIELD] = c
            s["language"] = lang
            s["id"] = s["url"]
    return snippets


def load_eval_dataset(dataset_name):
    try:
        eval_data = _load_data(dataset_name, EVAL_DATASETS)
    except Exception as e:
        # load snippet collection as evaluation data for code-only models
        # the description is used as the query and the corresponding snippet is the ground truth
        try:
            snippets = load_snippet_collection(dataset_name)
        except:
            raise e
        eval_data = [{"query": s[DESCRIPTION_FIELD], "relevant_ids": s["id"]} for s in snippets]
        
    golden_snippets = defaultdict(set)
    for instance in eval_data:
        golden_snippets[instance["query"]].update(instance["relevant_ids"])
    return tuple(golden_snippets.keys()), golden_snippets 


def load_train_dataset(dataset_name):
    return _load_data(dataset_name, TRAIN_DATASETS)


def make_gzipfile(file, overwrite=True):
    zipped_file = Path(file + ".gz")
    if zipped_file.exists() and not overwrite:
        return zipped_file
    with open(file, "rb") as f1:
        with gzip.open(zipped_file, "wb") as f2:
            shutil.copyfileobj(f1, f2)
    return zipped_file

def make_tarfile(source_dir, overwrite=True):
    source_dir = Path(source_dir)
    output_filename = source_dir.with_suffix(".tar.gz")
    if output_filename.exists() and not overwrite:
        return output_filename
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=".")
    return output_filename


def eval_datasets_from_regex(regex):
    return [ds for ds in EVAL_DATASETS if re.match(regex, ds)]