# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


import functools
import sys

import numpy as np

from codesearch.utils import Saveable
from codesearch.data_config import DESCRIPTION_FIELD, CODE_FIELD
    

class RetrievalModel(Saveable):
    
    def __init__(self):
        super().__init__()
        self._id2snippet = {}
        
    def query(self, query, n=10, projection=[], **kwargs):
        n = min(n, len(self._id2snippet))
        if projection and "id" not in projection:
            projection = ["id"] + projection
        sims = self.similarities(query, **kwargs)
        return_sims = "score" in projection or not projection
        if return_sims:
            ranking_ids, sims = self._sims_to_ranking(sims, n, return_sims)
        else:
            ranking_ids = self._sims_to_ranking(sims, n, return_sims)
            sims = None
        ranking_snippets = [dict(self._id2snippet[id_]) for id_ in ranking_ids]
        if sims:
            for s, sim in zip(ranking_snippets, sims):
                s["score"] = sim
        if projection:
            ranking_snippets = [ {f: r[f] for f in projection} for r in ranking_snippets]
       
        return ranking_snippets
    
    def log_query_results(self, queries, relevant_ids=[], projection=[DESCRIPTION_FIELD, CODE_FIELD], log_file=None):
        if log_file:
            stdout = sys.stdout
            with open(log_file, "a") as f:
                sys.stdout = f
                self._log_query_results(queries, relevant_ids, projection)
                sys.stdout = stdout
        else:
            self._log_query_results(queries, relevant_ids, projection)

    
    def _log_query_results(self, queries, relevant_ids=[], projection=[DESCRIPTION_FIELD, CODE_FIELD]):
        
        line1 = "*" * 40
        line2 = "-" * 40
        line3 = "-" * 10
        for q in queries:
            results = self.query(q, n=5, projection=projection)
            print(line1)
            print(f"QUERY: {q}")
            print(line1)
            print()
            for i, r in enumerate(results):
                annotation = ""
                if relevant_ids and r["id"] in relevant_ids:
                    annotation = " - RELEVANT"
                print(line2)
                print(f"RANK {i+1}{annotation}")
                print(line2)
                for f in projection:
                    if "\n" in str(r[f]):
                        print(f"{f.upper()}:")
                        print(r[f])
                    else:
                        print(f"{f.upper()}: {r[f]}")
                print(line2)
                print()
            print()
            print()
            
        
    
    def _sims_to_ranking(self, sims, n, return_sims):
        idxs = np.argpartition(sims, -n)[-n:]
        top_idxs = idxs[np.argsort(sims[idxs])][::-1]
        snippet_ids = [self.get_snippetid(top_idx) for top_idx in top_idxs]
        if return_sims:
            top_sims = [sims[top_idx] for top_idx in top_idxs]
            return snippet_ids, top_sims
        return snippet_ids
    
    def query_batch(self, queries, n=10, return_sims=False, **kwargs):
        sims_batch = self.similarities_batch(tuple(queries), **kwargs)
        snippet_ids_batch = []
        for sims in sims_batch:
            snippet_ids = self._sims_to_ranking(sims, n, return_sims)
            snippet_ids_batch.append(snippet_ids)
        return snippet_ids_batch
        
    def add_snippets(self, snippets):
        for s in snippets:
            self._id2snippet[s["id"]] = s
    
    def get_snippetid(self, idx):
        pass
            
    def similarities(self, query, **kwargs):
        pass
    
    def save(self, path):
        super().save(path)
    

class RetrievalEnsemble(RetrievalModel):
    
    def __init__(self, retrieval_models, weights):
        super().__init__()
        self.retrieval_models = retrieval_models
        self.weights = weights

    def similarities(self, query, model_kwargs=None):
        N = len(self.snippets)
        sims = np.zeros(shape=(N,), dtype=np.float32)
        for i, model in enumerate(self.retrieval_models):
            weight = self.weights[i] if self.weights else 1.
            sims += (weight * model.similarities(query))
        
        return sims            