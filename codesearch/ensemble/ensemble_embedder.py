# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================
 

from pathlib import Path
import numpy as np
import json

from codesearch.utils import Saveable, load_model
from codesearch.embedding_retrieval import RetrievalEmbedder

EPS = 10e-8

class EnsembleEmbedder(RetrievalEmbedder, Saveable):
    
    def __init__(self, embedder_names, weights=None):
        self._embedder_names = embedder_names
        self._embedders = [load_model(name) for name in embedder_names]
        self._weights = weights
        
    def embed_queries(self, queries):
        embs = []
        for embedder in self._embedders:
            embs_i = embedder.embed_queries(queries)
            embs_i /= np.linalg.norm(embs_i, keepdims=True, axis=1)
            embs.append(embs_i)
        return np.concatenate(embs, axis=1)
    
    def embed_snippets(self, snippets):
        embs = []
        for i, embedder in enumerate(self._embedders):
            embs_i = embedder.embed_snippets(snippets)
            embs_i /= np.linalg.norm(embs_i, keepdims=True, axis=1) + EPS
            if self._weights:
                embs_i *= self._weights[i]
            embs.append(embs_i)
        return np.concatenate(embs, axis=1)
        
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        with open(directory/"modelnames.json", "w") as f:
            json.dump(self._embedder_names, f)
        if self._weights:
            with open(directory/"weights.json", "w") as f:
                json.dump(self._weights, f)
                
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        modelnames_file = directory/f"modelnames.json"
        with open(directory/"modelnames.json") as f:
            modelnames = json.load(f)
       
        weights_file = directory/"weights.json"
        if weights_file.exists():
            with open(weights_file, "r") as f:
                weights = json.load(f)
        else:
            weights = None
            
        return cls(modelnames, weights)