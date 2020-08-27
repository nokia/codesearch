# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

"""
Evaluate a RetrievalModel or RetrievalEmbedder instance on annotated code search benchmarks.
"""

import os
from pathlib import Path
import json

from more_itertools import chunked
from tqdm import tqdm
import numpy as np

from codesearch.utils import load_model
from codesearch.embedding_retrieval import EmbeddingRetrievalModel, RetrievalEmbedder
from codesearch.data import load_snippet_collection
from codesearch.data_config import DESCRIPTION_FIELD, CODE_FIELD, CODESEARCHNET_LANGUAGES
from codesearch.evaluation import evaluate_and_dump, eval_mrr

def init_retrieval_model(model_path, snippet_collection):
    model = load_model(model_path)

    if isinstance(model, RetrievalEmbedder):
        model.batch_size = batch_size
        model = EmbeddingRetrievalModel(model)

    snippets = load_snippet_collection(snippet_collection)
    model.add_snippets(snippets)
    return model

def benchmark(model_path, snippet_collection, test_sets, model_name=None, output_dir=".", batch_size=512):
    if not Path(output_dir).exists():
        os.mkdir(output_dir)
    
    model = init_retrieval_model(model_path, snippet_collection)

    config = {"model_name": model_name if model_name else model_path}
    results = evaluate_and_dump(model, config, output_dir, None, test_sets)
    return results


def benchmark_on_pacs(model_path, model_name=None, output_dir=".", batch_size=512):
    pacs_datasets = [
        ("staqc-py-cleaned", ["staqc-py-raw-valid", "staqc-py-raw-test"] ),
        ("so-ds-feb20", ["so-ds-feb20-valid", "so-ds-feb20-test"]),
        ("conala-curated", ["conala-curated-0.5-test"])
    ]
    results = []
    for snippet_collection, test_datasets in tqdm(pacs_datasets):
        results_collection = benchmark(model_path, snippet_collection, test_datasets, model_name, output_dir, batch_size)
        results.extend(results_collection)
    return results


def benchmark_on_codesearchnet_distractors(model_path, model_name=None, output_dir=".", batch_size=1000, ds_type="valid"):
    """
    Replicate the evaluation with 999 random distractors as done in CodeSearchNet 
    """
    for lang in CODESEARCHNET_LANGUAGES:
        snippet_collection = f"codesearchnet-{lang}-{ds_type}"
        benchmark_with_distractors(model_path, snippet_collection, model_name, output_dir, batch_size)


def benchmark_with_distractors(model_path, snippet_collection, model_name=None, output_dir=".", batch_size=1000, log_samples=False):
    if not Path(output_dir).exists():
        os.mkdir(output_dir)
        
    model = load_model(model_path)
    snippets = np.array(load_snippet_collection(snippet_collection), dtype=np.object)
    # same seed as CodeSearchNet
    np.random.seed(0) 
    np.random.shuffle(snippets)
    mrr_sum = 0.
    num_batches = 0
    for snippets_batch in chunked(snippets, batch_size):
        
        if len(snippets_batch) < batch_size:
            break
        queries = [ s[DESCRIPTION_FIELD] for s in snippets_batch]
        snippets_batch_l = [s for s in snippets_batch]
        golden_snippet_ids = {s[DESCRIPTION_FIELD]: [s["id"]] for s in snippets_batch_l}
        if isinstance(model, RetrievalEmbedder):
            retrieval_model = EmbeddingRetrievalModel(model)
        else:
            retrieval_model = load_model(model_path)
        retrieval_model.add_snippets(snippets_batch_l) # we create a mini-retrieval model
        
        mrr_batch = eval_mrr(tuple(queries), golden_snippet_ids, retrieval_model, n=1000)
        mrr_sum += mrr_batch
        num_batches += 1
        if log_samples:
            retrieval_model.log_query_results(
                queries[:3], 
                relevant_ids=[s["id"] for s in snippets_batch[:3]], 
                projection=[DESCRIPTION_FIELD, CODE_FIELD, "score"]
            )
        
    mrr_score =  mrr_sum/num_batches
    
    record = {"mrr": mrr_score, "model_name": model_name or model_path}
    with open(Path(output_dir)/f"results-{snippet_collection}.jsonl", "a") as f:
        f.write(f"{json.dumps(record)}\n")
    return record