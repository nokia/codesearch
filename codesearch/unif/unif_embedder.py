# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import pdb 
from pathlib import Path

import numpy as np
from fasttext import load_model
import torch
from tqdm import trange

from codesearch.embedding_retrieval import RetrievalEmbedder
from codesearch.unif.unif_preprocessing import pad_sequences, extract_mask, fasttext_preprocess
from codesearch.unif.unif_modules import SimilarityModel
from codesearch.utils import Saveable
from codesearch.data_config import CODE_FIELD, LANGUAGE_FIELD


class UNIFEmbedder(RetrievalEmbedder, Saveable):
    
    def __init__(self, model, encoder, ft_model, batch_size=32, max_code_len=200, max_description_len=25):
        self.model = model
        self.encoder = encoder
        self.ft_model = ft_model
        self.batch_size = batch_size
        self.max_code_len = max_code_len
        self.max_description_len = max_description_len
        
    def encode_snippets(self, snippets):
        snippets_enc = []
        for snippet in snippets:
            code = snippet[CODE_FIELD]
            language = snippet[LANGUAGE_FIELD]
            snippet_enc = self.encoder.encode_code(code, language)[0]
            if not snippet_enc:
                snippet_enc = ["dummy"]
            snippets_enc.append(snippet_enc)
        return snippets_enc
        
    def embed_snippets(self, snippets):
        N = len(snippets)
        dim = self.ft_model.get_dimension()
        embeddings = np.zeros(shape=(N, dim), dtype=np.float32)
        for i in trange(0, N, self.batch_size, desc='Embedding snippets'):
            snippets_b = snippets[i: i + self.batch_size]
            snippets_enc = self.encode_snippets(snippets_b)
            snippets_enc = pad_sequences(snippets_enc, self.max_code_len)
            code_mask = extract_mask(snippets_enc)
            snippets_enc = fasttext_preprocess([t for s in snippets_enc for t in s], self.ft_model)
            embeddings_b = self.model.code_embedder(snippets_enc, code_mask)
            embeddings_b = embeddings_b.detach().cpu().numpy()
            embeddings[i:i+self.batch_size, :] = embeddings_b
        return embeddings
    
    def embed_queries(self, queries):
        N = len(queries)
        dim = self.ft_model.get_dimension()
        embeddings = np.zeros(shape=(N, dim), dtype=np.float32)
        for i in trange(0, N, self.batch_size, desc='Embedding queries'):
            queries_b = [self.encoder.encode_description(q) for q in queries[i:i+self.batch_size]]
            queries_b = pad_sequences(queries_b, self.max_description_len)
            query_mask = extract_mask(queries_b)
            queries_b = fasttext_preprocess([t for q in queries_b for t in q], self.ft_model)
            embeddings_b = self.model.description_embedder(queries_b, query_mask)
            embeddings_b = embeddings_b.detach().cpu().numpy()
            embeddings[i: i + self.batch_size, :] = embeddings_b
        return embeddings
    
    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        self.ft_model.save_model(str(directory/"fasttext_model.bin"))
        torch.save(self.model.state_dict(), str(directory/"similarity_model_state"))
        self.encoder.save(str(directory/"encoder"))
        
    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        ft_model = load_model(str(directory/"fasttext_model.bin"))
        sim_model = SimilarityModel(ft_model)
        sim_model.load_state_dict(torch.load(str(directory/"similarity_model_state"), map_location=torch.device('cpu')))
        enc = Saveable.load(str(directory/"encoder"))
        
        return cls(sim_model, enc, ft_model)
    
