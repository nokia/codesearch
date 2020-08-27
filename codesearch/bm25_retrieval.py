# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================


from pathlib import Path
import functools
import json

import pickle 
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from codesearch.retrieval import RetrievalModel
from codesearch.data_config import CODE_FIELD, LANGUAGE_FIELD, DESCRIPTION_FIELD
from codesearch.utils import Saveable

class BM250RetrievalModel(RetrievalModel, Saveable):
    
    def __init__(self, encoder, index_code=True, index_description=False):
        super().__init__()
        self._snippets = []
        self.bm250_model = None
        self.encoder = encoder
        self.index_code = index_code
        self.index_description = index_description
        
    def get_snippetid(self, idx):
        return self._snippets[idx]["id"]
    
    @functools.lru_cache(maxsize=5000)
    def similarities(self, query):
        query = self.encoder.encode_description(query)
        return self.bm250_model.get_scores(query)
    
    def add_snippets(self, snippets):
        super().add_snippets(snippets)
        self._snippets.extend(snippets)
        encoded_snippets = self.encode_snippets(self._snippets)
        self.bm250_model = BM25Okapi(encoded_snippets)
        
    def encode_snippets(self, snippets):
        encoded_snippets = []
        for i, s in enumerate(tqdm(snippets, desc="Preprocessing snippets")):
            s_enc = []
            if self.index_code:
                s_enc.extend(self.encoder.encode_code(s[CODE_FIELD], s[LANGUAGE_FIELD])[0])
            if self.index_description:
                s_enc.extend(self.encoder.encode_description(s[DESCRIPTION_FIELD]))
            encoded_snippets.append(s_enc)
        return encoded_snippets
    
    def save(self, path):
        super().save(path)
        path = Path(path)
        self.encoder.save(path/"encoder")
        with open(path/"kwargs.json", "w") as f:
            json.dump({"index_code": self.index_code, "index_description": self.index_description}, f)
    
    @classmethod
    def load(cls, path):
        path = Path(path)
        encoder = Saveable.load(path/"encoder")
        with open(path/"kwargs.json") as f:
            kwargs = json.load(f)
        return cls(encoder, **kwargs)
    
    @classmethod
    def from_snippets(cls, snippets, encoder, index_code=True, index_description=False):
        model = cls(encoder, index_code, index_description)
        model.add_snippets(snippets)
        return model