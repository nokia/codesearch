# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import os
from pathlib import Path
import json
from copy import deepcopy

CODE_FIELD = os.environ.get("CS_CODE_FIELD", "code")
DESCRIPTION_FIELD = os.environ.get("CS_DESCRIPTION_FIELD", "description")
LANGUAGE_FIELD = os.environ.get("CS_LANGUAGE_FIELD", "language")

BASE_URL = "https://zenodo.org/record/4001602/files/{}?download=1"
MODEL_BASE_URL = BASE_URL 
DATASET_BASE_URL = BASE_URL

MODULE_DIR = Path(__file__).parent.absolute()

DATA_DIR = Path(os.environ.get("CODE_SEARCH_DATA_DIR", str(MODULE_DIR/"data")))

if not DATA_DIR.exists():
    DATA_DIR.mkdir()


DATASETS_DIR = DATA_DIR/"datasets"
if not DATASETS_DIR.exists():
    DATASETS_DIR.mkdir()

MODELS_DIR = DATA_DIR/"pretrained-models"
if not MODELS_DIR.exists():
    MODELS_DIR.mkdir()

    
# We specify the provided datasets and pre-trained models
SNIPPET_COLLECTIONS = {
    "so-ds-feb20": {
        "path": DATASETS_DIR/"so-ds-feb20.jsonl",
        "url": DATASET_BASE_URL.format("so-ds-feb20.jsonl.gz")},
    "staqc-py-cleaned": {
        "path": DATASETS_DIR/"staqc-py-cleaned.jsonl",
        "url": DATASET_BASE_URL.format("staqc-py-cleaned.jsonl.gz")
    },
    "conala-curated": {
        "path": DATASETS_DIR/"conala-curated-snippets.jsonl",
        "url": DATASET_BASE_URL.format("conala-curated-snippets.jsonl.gz")
    },
    "codesearchnet-java": {
        "path": DATASETS_DIR/"codesearchnet/java_dedupe_definitions_v2.pkl",
        "url": "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip",
        "extract_at": DATASETS_DIR/"codesearchnet"
    },
    "codesearchnet-go": {
        "path": DATASETS_DIR/"codesearchnet/go_dedupe_definitions_v2.pkl",
        "url": "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip",
        "extract_at": DATASETS_DIR/"codesearchnet"
    },
    "codesearchnet-ruby": {
        "path": DATASETS_DIR/"codesearchnet/ruby_dedupe_definitions_v2.pkl",
        "url": "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip",
        "extract_at": DATASETS_DIR/"codesearchnet"
    },
     "codesearchnet-java-train": { # snippet collections are training/validation/test data for code-only models
        "path_pattern": DATASETS_DIR/"codesearchnet/java/final/jsonl/train/*.jsonl.gz",
        "url": "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip",
        "extract_at": DATASETS_DIR/"codesearchnet"
    },
    "codesearchnet-java-valid": {
        "path_pattern": DATASETS_DIR/"codesearchnet/java/final/jsonl/valid/*.jsonl.gz",
        "url": "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip",
        "extract_at": DATASETS_DIR/"codesearchnet"
    },
    "codesearchnet-java-test": {
        "path_pattern": DATASETS_DIR/"codesearchnet/java/final/jsonl/test/*.jsonl.gz",
        "url": "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip",
        "extract_at": DATASETS_DIR/"codesearchnet"
    }
    
}

def _add_codesearchnet_datasets(language):
    url = f"https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip"
    extract_at = DATASETS_DIR/"codesearchnet"
    path_all = DATASETS_DIR/f"codesearchnet/{language}_dedupe_definitions_v2.pkl"
    SNIPPET_COLLECTIONS[f"codesearchnet-{language}"] = {
        "path": path_all,
        "url": url,
        "extract_at": extract_at
    }
    for ds_type in ["train", "valid", "test"]:
        SNIPPET_COLLECTIONS[f"codesearchnet-{language}-{ds_type}"] = {
            "path_pattern": DATASETS_DIR/f"codesearchnet/{language}/final/jsonl/{ds_type}/*.jsonl.gz",
            "url": url,
            "extract_at": extract_at
        }

CODESEARCHNET_LANGUAGES = ["go", "java", "javascript", "php", "python", "ruby"]
for lang in CODESEARCHNET_LANGUAGES:
    _add_codesearchnet_datasets(lang)

EVAL_DATASETS = {
    "so-ds-feb20-valid": { 
        "path": DATASETS_DIR/"so-ds-feb20-valid-pacsv1.jsonl", 
        "url": DATASET_BASE_URL.format("so-ds-feb20-valid-pacsv1.jsonl.gz")},
    "so-ds-feb20-test": { 
        "path": DATASETS_DIR/"so-ds-feb20-test.jsonl",
        "url": DATASET_BASE_URL.format("so-ds-feb20-test.jsonl.gz")},
    "staqc-py-raw-test": {
        "path": DATASETS_DIR/"staqc-py-test-raw.jsonl",
        "url": DATASET_BASE_URL.format("staqc-py-test-raw.jsonl.gz")
    },
    "staqc-py-raw-valid": {
        "path": DATASETS_DIR/"staqc-py-valid-raw-pacsv1.jsonl",
        "url": DATASET_BASE_URL.format("staqc-py-valid-raw-pacsv1.jsonl.gz")
    },
    "conala-curated-0.5-test": {
        "path": DATASETS_DIR/"conala-test-curated-0.5.jsonl",
        "url": DATASET_BASE_URL.format("conala-test-curated-0.5.jsonl.gz")
    }
}

TRAIN_DATASETS = {
    "so-duplicates-feb20": {        # this dataset has some overlap with the PACS evaluation data
                                    # don't use this to evaluate on PACS
        "path": DATASETS_DIR/"so-all-duplicates-feb20.jsonl",
        "url": DATASET_BASE_URL.format("so-duplicates-feb20.jsonl.gz")},
    "so-duplicates-pacs-train": {    # this dataset has no overlap with the PACS evaluation data
        "path": DATASETS_DIR/"so-duplicates-pacsv1-train.jsonl",
        "url": DATASET_BASE_URL.format("so-duplicates-pacsv1-train.jsonl.gz")},   
    "so-python-question-titles-feb20": {
        "path": DATASETS_DIR/"so-python-question-titles-feb20.txt",
        "url": DATASET_BASE_URL.format("so-python-question-titles-feb20.txt.gz")},
}

MODELS = {
    "use-embedder-pacs": {
        "path": MODELS_DIR/"use-embedder-pacsv1",
        "url": DATASET_BASE_URL.format("use-embedder-pacsv1.tar.gz")
    },
    "ensemble-embedder-pacs": {
        "path": MODELS_DIR/"ensemble-embedder-pacsv1",
        "url": DATASET_BASE_URL.format("ensemble-embedder-pacsv1.tar.gz")
    },
    "ncs-embedder-so-ds-feb20": { 
        "path": MODELS_DIR/"ncs-embedder-so.feb20",
        "url": DATASET_BASE_URL.format("ncs-embedder-so.tar.gz")
    },
    "ncs-embedder-staqc-py": {
        "path": MODELS_DIR/"ncs-embedder-staqc-py",
        "url": DATASET_BASE_URL.format("ncs-embedder-staqc-py.tar.gz")
    },
    "tnbow-embedder-so-ds-feb20": {
        "path": MODELS_DIR/"tnbow-embedder-so.feb20",
        "url": DATASET_BASE_URL.format("tnbow-embedder-so.tar.gz")
    }
}


DATA_REGISTRY = {
    "SNIPPET_COLLECTIONS": SNIPPET_COLLECTIONS,
    "TRAIN_DATASETS": TRAIN_DATASETS,
    "EVAL_DATASETS": EVAL_DATASETS,
    "MODELS": MODELS
    }

# Custom data/pre-trained models can be saved in data_registry.json
DATA_REGISTRY_FILE = DATA_DIR/"data_registry.json"

if DATA_REGISTRY_FILE.exists():
    with open(DATA_REGISTRY_FILE) as f:
        CUSTOM_DATA_REGISTRY = json.load(f)
        for data_registry in CUSTOM_DATA_REGISTRY.values():
            for data_spec in data_registry.values():
                data_spec["path"] = Path(data_spec["path"])
else:
    CUSTOM_DATA_REGISTRY = {
        "SNIPPET_COLLECTIONS": {},
        "TRAIN_DATASETS": {},
        "EVAL_DATASETS": {},
        "MODELS": {}
    }

for data_type, data_registry in CUSTOM_DATA_REGISTRY.items():
    for name, data_spec in data_registry.items():
        if data_type in DATA_REGISTRY:
            DATA_REGISTRY[data_type][name] = data_spec
        else:
            print(f"Custom data registry entry {data_spec} with name {name} has an invalid type. type should be {sorted(DATA_REGISTRY.keys())}")


def save_data_registry():
    with open(DATA_REGISTRY_FILE, "w") as f:
        # Convert PosixPaths to strings before dumping to json
        obj = deepcopy(CUSTOM_DATA_REGISTRY)
        for data_registry in obj.values():
            for data_spec in data_registry.values():
                data_spec["path"] = str(data_spec["path"])
        json.dump(obj, f)

def _data_entry(name):
    for registry in DATA_REGISTRY.values():
        if name in DATA_REGISTRY[registry]:
            return 

def _register_data(data_type, name, filename, url):
    base_dir = MODELS_DIR if data_type == "MODELS" else DATASETS_DIR
    model_spec = {"path": base_dir/filename, "url": url }

    if name in DATA_REGISTRY[data_type]:
        if model_spec != DATA_REGISTRY[data_type][name]:
            raise ValueError(f"There is already an entry with name {name}.")
        return
    DATA_REGISTRY[data_type][name] = model_spec
    CUSTOM_DATA_REGISTRY[data_type][name] = dict(model_spec)
    save_data_registry()


def register_model(name, filename, url):
    """
    Register a pre-trained model. 
    
    The pre-trained model should either be available at the given url or
    present under the `MODELS_DIR` directory.

    Registered models can be downloaded (if an URL specified) and loaded easily by referring to their names.
    """
    _register_data("MODELS", name, filename, url)

def register_training_dataset(name, filename, url):
    _register_data("TRAIN_DATASETS", name, filename, url)

def register_eval_dataset(name, filename, url):
    _register_data("EVAL_DATSETS", name, filename, url)

def register_snippet_collection(name, filename, url):
    _register_data("SNIPPET_COLLECTIONS", name, filename, url)