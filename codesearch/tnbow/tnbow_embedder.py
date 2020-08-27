# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from pathlib import Path

from codesearch.utils import Saveable
from codesearch.embedding_retrieval import RetrievalEmbedder
from codesearch.sequence_embedding import BoWEmbedder
from codesearch.snippet_embedding import SnippetTitleEmbedder

class TnbowEmbedder(RetrievalEmbedder, Saveable):
    
    def __init__(self, nbow_embedder):
        self._nbow_embedder = nbow_embedder
        self._snippet_embedder = SnippetTitleEmbedder(self._nbow_embedder)
        
    def embed_queries(self, queries):
        return self._nbow_embedder(queries)
    
    def embed_snippets(self, snippets):
        return self._snippet_embedder(snippets)
    
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        self._nbow_embedder.save(directory/"nbow_embedder")
        
    @classmethod
    def from_fasttext_model(cls, ft_model, encoder):
        nbow_embedder = BoWEmbedder(encoder, ft_model)
        return cls(nbow_embedder)
        
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        nbow_embedder = Saveable.load(directory/"nbow_embedder")
        return cls(nbow_embedder)
        