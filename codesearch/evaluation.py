# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import json
import copy
import functools
from functools import partial
from collections import defaultdict

from codesearch.embedding_retrieval import  EmbeddingRetrievalModel
from codesearch.utils import progress_generate
from codesearch.data import load_eval_dataset


def mrr(retrieved_snippets, golden_snippets):
    rranks = reciprocal_ranks(retrieved_snippets, golden_snippets)
    return sum(rranks)/len(rranks)

def reciprocal_rank(retrieved, golden):
    if not retrieved: return 0
    for i, s in enumerate(retrieved):
        if s in golden:
            break
    return 1/(i + 1) if s in golden else 0

def reciprocal_ranks(retrieved_snippets, golden_snippets):
    rranks = []
    for golden, retrieved in zip(golden_snippets, retrieved_snippets):
        rranks.append(reciprocal_rank(retrieved, golden))
    return rranks

def recall(retrieved_snippets, golden_snippets):
    num_retrieved = 0
    for retrieved_snippets_, golden_snippets_ in zip(retrieved_snippets, golden_snippets):
        num_retrieved += int(any(golden in retrieved_snippets_ for golden in golden_snippets_))
    return num_retrieved/len(golden_snippets)
        

@functools.lru_cache()
def retrieve_snippets(queries, retrieval_model, n=10):
    if isinstance(retrieval_model, EmbeddingRetrievalModel):
        return retrieval_model.query_batch(queries, n=n)
    retrieved_snippet_ids = []
    for query in progress_generate(queries):
        r = [ s["id"] for s in retrieval_model.query(query, projection=["id"], n=n)]
        retrieved_snippet_ids.append(r)
    return retrieved_snippet_ids

def eval_mrr(queries, golden_snippet_ids, retrieval_model, n=10):
    retrieved_snippet_ids = retrieve_snippets(queries, retrieval_model, n)
    golden_snippet_ids = [golden_snippet_ids[query] for query in queries]
    return mrr(retrieved_snippet_ids, golden_snippet_ids)


def eval_recall(queries, golden_snippet_ids, retrieval_model, n=10):    
    retrieved_snippet_ids = retrieve_snippets(queries, retrieval_model, n)
    golden_snippet_ids = [golden_snippet_ids[query] for query in queries]
    return recall(retrieved_snippet_ids, golden_snippet_ids)


def title_as_query(snippets):
    queries = [] 
    golden_snippet_ids = defaultdict(list)
    for s in snippets:
        query = s["title"]
        queries.append(query)
        golden_snippet_ids[query].append(s["globalid"])
    return tuple(set(queries)), golden_snippet_ids

def all_duplicates(snippets):
    queries = []
    golden_snippet_ids = defaultdict(list)
    for s in snippets:
        if "duplicates" in s:
            for _, duplicate_title in s["duplicates"]:
                queries.append(duplicate_title)
                golden_snippet_ids[duplicate_title].append(s["globalid"])
    return tuple(set(queries)), golden_snippet_ids

def first_duplicates(snippets):
    queries = []
    golden_snippet_ids = defaultdict(list)
    for s in snippets:
        if "duplicates" in s:
            query = s["duplicates"][0][1]
            queries.append(query)
            golden_snippet_ids[query].append(s["globalid"])
    print(len(set(queries)))
    return tuple(set(queries)), golden_snippet_ids
       
def evaluate(retrieval_model, dataset_name=None, dataset=None, metrics=["mrr", "recall@3", "recall@10"]):
    results = {}
    if not dataset:
        if not dataset_name: 
            raise ValueError("Either provide a dataset_name or dataset")
        dataset = load_eval_dataset(dataset_name)
    queries, golden_snippet_ids = dataset
    for metric in metrics:
        metric_fn = METRICS[metric]
        results[metric] = metric_fn(queries, golden_snippet_ids, retrieval_model)
    return results
    
    
METRICS = {"mrr": partial(eval_mrr, n=20), 
           "recall@3": partial(eval_recall, n=3),
           "recall@10": partial(eval_recall, n=10)
          }


def evaluate_and_dump_(retrieval_model, config, results_fn, dataset_name=None, dataset=None):
    results = evaluate(retrieval_model, dataset_name=dataset_name, dataset=dataset)
    with open(results_fn, "a") as f:
        record = copy.deepcopy(config)
        record.update(results)
        f.write(f"{json.dumps(record)}\n")
    return results


def evaluate_and_dump(retrieval_model, config, output_dir, valid_dataset, test_datasets, sample_queries=[]):
    datasets = list(test_datasets)
    if valid_dataset:
        datasets.append(valid_dataset)
    results = {}
    for ds in datasets:
        results[ds] = evaluate_and_dump_(retrieval_model, config, f"{output_dir}/results.{ds}.jsonl", ds)
    if sample_queries:
        retrieval_model.log_query_results(sample_queries)
    return results