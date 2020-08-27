# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from pathlib import Path
import dill
import numpy as np
import functools
import logging

from tqdm import tqdm

from codesearch.retrieval import RetrievalModel
from codesearch.utils import Saveable
from codesearch.snippet_embedding import SnippetTitleEmbedder

logger = logging.getLogger(__name__)

EPS = 10e-8

class RetrievalEmbedder(object):
    """ Embeds queries and snippets in the same vector space."""

    def embed_queries(self, queries):
        pass
    
    def embed_snippets(self, snippets):
        pass
    
def nan_logger(embedder):
    def nan_logger_fn(*args, **kwargs):
        embs = embedder(*args, **kwargs)
        if np.isnan(embs).any():
            print("Embedder produced NaN values")
        if len(np.where(~embs.any(axis=1))[0]) > 0:
            print("Contains all zero rows")
        return embs
    return nan_logger_fn

class EmbeddingRetrievalModel(RetrievalModel):
    
    def __init__(self, retrieval_embedder):
        super().__init__()
        self._snippetids = [] 
        self._snippetid2idx = {} # snippet id to embedding index
        self._unused_array_rows = 0
        self._embs = None
        self._embedder = retrieval_embedder
    
    @functools.lru_cache(maxsize=5000)
    def similarities(self, query):
        query = self._embedder.embed_queries([query])[0]
        query = query/np.linalg.norm(query)
        sims = np.dot(self._embs, query)
        return sims
    
    @functools.lru_cache(maxsize=5000)
    def similarities_batch(self, queries):
        # N x dim
        queries = self._embedder.embed_queries(queries)
        queries = queries/np.linalg.norm(queries)
        # N x M
        sims = np.dot(queries, self._embs.T)
        return sims
    
    
    def get_snippetid(self, idx):
        return self._snippetids[idx]
    
    def add_snippets(self, snippets):
        super().add_snippets(snippets)
        embs = self._embedder.embed_snippets(snippets)
        norms = np.linalg.norm(embs, keepdims=True, axis=1) + EPS
        embs /= norms
        idxs = []
        num_new = 0
        for s in snippets:
            id_ = s["id"]
            if id_ not in self._snippetid2idx:
                num_new += 1
                self._snippetid2idx[id_] = len(self._snippetids)
                self._snippetids.append(id_)
            idxs.append(self._snippetid2idx[id_])
        self._grow_embeddings(num_new, embs.shape[1])
        self._update_embeddings(idxs, embs)
        
    def _grow_embeddings(self, N, dim):
        new_embs = np.zeros(shape=(N,dim), dtype=np.float32)
        if self._embs is None:
            self._embs = new_embs
        else:
            self._embs = np.concatenate([self._embs, new_embs], axis=0)
                
    def _update_embeddings(self, idxs, embeddings):
        self._embs[idxs, :] = embeddings    
    
    def save(self, path):
        super().save(path)
        path = Path(path)
        np.save(path/'embs.npy', self._embs)
        self._embedder.save(path/"retrieval_embedder")
        with open(path/"index", "wb") as f:
            dill.dump((self._snippetids, self._snippetid2idx, self._unused_array_rows), f)   
        
    @classmethod
    def load(cls, path):
        path = Path(path)
        embs = np.load(path/'embs.npy')
        embedder = Saveable.load(path/"retrieval_embedder")
        with open(path/"index", "rb") as f:
            snippetids, snippetid2idx, unused_array_rows = dill.load(f)
        
        obj = cls(embedder)
        obj._embs = embs
        obj._snippetids = snippetids
        obj._snippetid2dix = snippetid2idx
        obj._unused_array_rows = unused_array_rows
    
        return obj